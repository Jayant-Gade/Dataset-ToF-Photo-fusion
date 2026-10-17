import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_augmented_dataset" if os.path.isdir("gesturanet_augmented_dataset") else "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"    # Retained folder name; calibrated to 30 cm
WEIGHTS_PATH = "weights/gesturanet_4class_best.pth"

ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign"
]

# Mode Toggle: Set True to load precomputed .npy files from disk; False to synthesize live
USE_DISK_NPY = True

# UI Dimensions
VIEWPORT_SIZE = 640
SIDEBAR_WIDTH = 400
WINDOW_NAME = "GesturaNet - Validation & Depth Sensitivity Inspector (30cm / Null BG)"

# Sensor Simulation Parameters (30 cm scale)
DEFAULT_DISTANCE_CM = 30.0     # 30.0 cm baseline
MIN_DISTANCE_CM = 15.0         # Minimum tuning limit
MAX_DISTANCE_CM = 50.0         # Maximum tuning limit
DISTANCE_STEP_CM = 0.2         # Tuning increment per keypress
NULL_VALUE_M = 0.000           # Background / Open air
NOISE_PERCENT = 0.015          # 1.5% sensor noise (~4.5 mm at 30 cm)
ZONE_ACTIVATION_THRESHOLD = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class TwoStreamGesturaNet(nn.Module):
    """Multimodal fusion network matching the trained weights."""
    def __init__(self, num_classes=4):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_rgb = mobilenet_v3_small(weights=weights)
        self.rgb_features = base_rgb.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(576 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, rgb, depth):
        x_rgb = self.rgb_pool(self.rgb_features(rgb)).flatten(1)
        x_depth = self.depth_branch(depth)
        fused = torch.cat((x_rgb, x_depth), dim=1)
        return self.classifier(fused)


def extract_robust_hand_mask(image_bgr):
    """
    Skin segmentation using YCrCb Chromaticity (Cr - Cb) and HSV saturation.
    Prevents background whiteouts on off-white walls.
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    skin_diff = cr - cb
    skin_mask = np.where(skin_diff > 12, 255, 0).astype(np.uint8)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 20, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_and(skin_mask, sat_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned)

    h, w = image_bgr.shape[:2]
    if contours:
        valid_cnts = [c for c in contours if 400 < cv2.contourArea(c) < (h * w * 0.85)]
        if valid_cnts:
            largest = max(valid_cnts, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            return mask

    # Inverted Otsu luminance fallback
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cleaned_otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours_otsu, _ = cv2.findContours(cleaned_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours_otsu:
        valid_cnts = [c for c in contours_otsu if 400 < cv2.contourArea(c) < (h * w * 0.85)]
        if valid_cnts:
            largest = max(valid_cnts, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def synthesize_tof_array(hand_mask, dist_cm):
    """
    Computes an 8x8 depth array using tuned distance (dist_cm):
    - Hand zones receive physical distance + curvature + 1.5% noise.
    - Background zones receive NULL_VALUE_M (0.0).
    """
    hand_dist_m = dist_cm / 100.0
    h, w = hand_mask.shape
    cell_h, cell_w = h // 8, w // 8

    mask_float = (hand_mask > 0).astype(np.float32)
    coverage_8x8 = cv2.resize(mask_float, (8, 8), interpolation=cv2.INTER_AREA)

    # Curvature calculation: center of hand projects slightly forward
    if np.count_nonzero(hand_mask) > 200:
        dist_transform = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
        dt_8x8 = cv2.resize(dist_transform, (8, 8), interpolation=cv2.INTER_AREA)
        max_dt = float(dt_8x8.max())
        norm_dt = (dt_8x8 / max_dt) if max_dt > 0.001 else np.zeros((8, 8), dtype=np.float32)
        hand_depth_8x8 = (hand_dist_m + 0.010) - (norm_dt * 0.020)
    else:
        hand_depth_8x8 = np.full((8, 8), hand_dist_m, dtype=np.float32)

    depth_matrix = np.full((8, 8), NULL_VALUE_M, dtype=np.float32)

    for r in range(8):
        for c in range(8):
            if coverage_8x8[r, c] >= ZONE_ACTIVATION_THRESHOLD:
                base_d = hand_depth_8x8[r, c]
                noise = np.random.normal(0.0, base_d * NOISE_PERCENT)
                depth_matrix[r, c] = np.clip(base_d + noise, 0.10, 1.00)
            else:
                depth_matrix[r, c] = NULL_VALUE_M

    return np.nan_to_num(depth_matrix, nan=NULL_VALUE_M).astype(np.float32)


def draw_8x8_overlay(image):
    overlay = image.copy()
    step = VIEWPORT_SIZE // 8
    for i in range(1, 8):
        pos = i * step
        cv2.line(overlay, (pos, 0), (pos, VIEWPORT_SIZE), (0, 255, 0), 1)
        cv2.line(overlay, (0, pos), (VIEWPORT_SIZE, pos), (0, 255, 0), 1)
    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


def create_heatmap_view(depth_matrix, size=150):
    """
    Renders an 8x8 heatmap for the sidebar:
    - Active cells: Color-coded with distance values.
    - Null cells: Dark background with '-'.
    """
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    step = size // 8

    active_mask = depth_matrix > 0.0
    if np.any(active_mask):
        active_vals = depth_matrix[active_mask]
        min_v = max(0.15, active_vals.min() - 0.02)
        max_v = min_v + 0.05
        norm = np.clip((depth_matrix - min_v) / (max_v - min_v + 1e-5), 0.0, 1.0)
        heat = (255 * (1.0 - norm)).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
        colored_resized = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)

        for r in range(8):
            for c in range(8):
                if depth_matrix[r, c] > 0.0:
                    canvas[r * step:(r + 1) * step, c * step:(c + 1) * step] = \
                        colored_resized[r * step:(r + 1) * step, c * step:(c + 1) * step]

    for r in range(8):
        for c in range(8):
            val_m = depth_matrix[r, c]
            tx = c * step + int(step * 0.10)
            ty = r * step + int(step * 0.65)

            if val_m > 0.0:
                txt = f"{val_m * 100:.0f}"
                txt_color = (0, 0, 0) if val_m < 0.305 else (255, 255, 255)
            else:
                txt = "-"
                txt_color = (90, 90, 90)

            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.33, txt_color, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (c * step, r * step), ((c + 1) * step, (r + 1) * step), (55, 55, 55), 1)

    return canvas


def load_dataset_samples():
    """Loads dataset samples grouped chronologically by gesture folder."""
    dataset_records = []
    for g_idx, gesture in enumerate(ACTIVE_GESTURES):
        folder = os.path.join(BASE_DATASET_DIR, gesture, TARGET_DISTANCE_DIR)
        files = sorted(glob.glob(os.path.join(folder, "img_*.jpg")))
        for f in files:
            dataset_records.append({"path": f, "gesture": gesture, "g_idx": g_idx})
    return dataset_records


def find_gesture_jump(samples, current_idx, direction):
    """Finds the first index of the next or previous gesture group."""
    total = len(samples)
    current_g_idx = samples[current_idx]["g_idx"]
    target_g_idx = (current_g_idx + direction) % len(ACTIVE_GESTURES)

    for i, item in enumerate(samples):
        if item["g_idx"] == target_g_idx:
            return i
    return (current_idx + direction) % total


def main():
    global USE_DISK_NPY
    samples = load_dataset_samples()
    if not samples:
        print(f"No captured images found in '{BASE_DATASET_DIR}' across classes: {ACTIVE_GESTURES}")
        return

    print(f"Loaded {len(samples)} images from '{BASE_DATASET_DIR}'.")

    # Load Model
    model = TwoStreamGesturaNet(num_classes=len(ACTIVE_GESTURES)).to(DEVICE)
    if os.path.isfile(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded trained weights from: {WEIGHTS_PATH}")
    else:
        print(f"Warning: {WEIGHTS_PATH} not found. Running with uninitialized weights.")
    model.eval()

    rgb_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    current_idx = 0
    current_distance_cm = DEFAULT_DISTANCE_CM
    canvas = np.zeros((VIEWPORT_SIZE, VIEWPORT_SIZE + SIDEBAR_WIDTH, 3), dtype=np.uint8)
    cached_mask = None
    last_loaded_idx = -1

    while True:
        record = samples[current_idx]
        img_path = record["path"]
        ground_truth = record["gesture"]
        g_num = record["g_idx"] + 1

        # Cache mask computation unless image index changes
        if current_idx != last_loaded_idx:
            raw_bgr = cv2.imread(img_path)
            if raw_bgr.shape[0] != VIEWPORT_SIZE or raw_bgr.shape[1] != VIEWPORT_SIZE:
                raw_bgr = cv2.resize(raw_bgr, (VIEWPORT_SIZE, VIEWPORT_SIZE))
            cached_mask = extract_robust_hand_mask(raw_bgr)
            last_loaded_idx = current_idx

        # =========================================================================
        # 1. ToF Depth Selection (Option 1a vs. Option 1b)
        # =========================================================================

        # [Option 1a: Realtime synthesized 8x8 ToF generation with distance tuning]
        # depth_8x8 = synthesize_tof_array(cached_mask, current_distance_cm)

        # [Option 1b: Load precomputed 8x8 ToF from disk (.npy) with fallback]
        base_no_ext = os.path.splitext(img_path)[0]
        saved_tof_path = f"{base_no_ext}_tof.npy"

        if USE_DISK_NPY and os.path.isfile(saved_tof_path):
            raw_loaded = np.load(saved_tof_path).astype(np.float32)
            depth_8x8 = np.nan_to_num(raw_loaded, nan=0.0, posinf=0.0, neginf=0.0)
            tof_source_label = "DISK (.npy)"
        else:
            depth_8x8 = synthesize_tof_array(cached_mask, current_distance_cm)
            tof_source_label = "SYNTHESIZED"

        # =========================================================================
        # 2. Run Model Inference
        # =========================================================================
        pil_img = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        rgb_tensor = rgb_transforms(pil_img).unsqueeze(0).to(DEVICE)
        depth_tensor = torch.from_numpy(depth_8x8).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(rgb_tensor, depth_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        results = [
            {"gesture": ACTIVE_GESTURES[i], "score": float(probs[i])}
            for i in range(len(ACTIVE_GESTURES))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        top_prediction = results[0]["gesture"]
        is_correct = (top_prediction == ground_truth)

        # =========================================================================
        # 3. Render Viewport (Left)
        # =========================================================================
        viewport = draw_8x8_overlay(raw_bgr)
        canvas[:, :VIEWPORT_SIZE] = viewport

        # =========================================================================
        # 4. Render Dashboard Sidebar (Right)
        # =========================================================================
        sidebar = canvas[:, VIEWPORT_SIZE:]
        sidebar[:] = (26, 28, 32)

        # Header Section
        cv2.putText(sidebar, "VALIDATION BENCH", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 42), (SIDEBAR_WIDTH - 20, 42), (65, 65, 65), 1)

        # Sample Info & Active Folder
        fname = os.path.basename(img_path)
        cv2.putText(sidebar, f"Folder [{g_num}/{len(ACTIVE_GESTURES)}]: {ground_truth.upper()}", (20, 68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 200), 1)
        cv2.putText(sidebar, f"File [{current_idx + 1}/{len(samples)}]: {fname}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Depth Source & Distance Dial Card
        cv2.rectangle(sidebar, (18, 106), (SIDEBAR_WIDTH - 18, 172), (38, 42, 48), -1)
        cv2.rectangle(sidebar, (18, 106), (SIDEBAR_WIDTH - 18, 172), (70, 75, 82), 1)

        source_color = (0, 255, 120) if "DISK" in tof_source_label else (0, 200, 255)
        cv2.putText(sidebar, f"ToF Source: {tof_source_label}", (28, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, source_color, 1)

        if "DISK" in tof_source_label:
            cv2.putText(sidebar, "Mode: Precomputed file on disk", (28, 148),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(sidebar, "[T] Key: Toggle Live Synthesis", (28, 164),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        else:
            cv2.putText(sidebar, f"Tuned Depth: {current_distance_cm:.1f} cm", (28, 148),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)
            cv2.putText(sidebar, "UP / DOWN: Tune | [T]: Toggle Disk", (28, 164),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Prediction Confidence Card
        cv2.rectangle(sidebar, (18, 182), (SIDEBAR_WIDTH - 18, 380), (32, 35, 40), -1)
        cv2.rectangle(sidebar, (18, 182), (SIDEBAR_WIDTH - 18, 380), (60, 65, 72), 1)

        verdict_color = (0, 210, 0) if is_correct else (0, 60, 230)
        verdict_text = "MATCH CORRECT" if is_correct else "MISMATCH"
        cv2.putText(sidebar, f"Top: {top_prediction.upper()}", (28, 206),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, verdict_color, 1)
        cv2.putText(sidebar, f"[{verdict_text}]", (28, 226),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, verdict_color, 1)

        # Class Probability Bars
        bar_start_y = 250
        for i, item in enumerate(results):
            g_name = item["gesture"]
            score = item["score"]
            y_pos = bar_start_y + (i * 30)

            txt = f"{g_name:<11} {score * 100:5.1f}%"
            txt_color = (255, 255, 255) if i == 0 else (170, 170, 170)
            cv2.putText(sidebar, txt, (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_color, 1)

            bar_x = 185
            bar_w = 175
            cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + bar_w, y_pos - 2), (50, 52, 58), -1)

            fill_w = int(bar_w * score)
            fill_color = (0, 200, 120) if i == 0 else (120, 120, 120)
            if fill_w > 0:
                cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + fill_w, y_pos - 2), fill_color, -1)

        # 8x8 ToF Heatmap Preview
        cv2.putText(sidebar, "8x8 ToF Depth Map (0.0 = Null)", (20, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
        heatmap_view = create_heatmap_view(depth_8x8, size=150)
        sidebar[418:418 + 150, 20:20 + 150] = heatmap_view

        active_cnt = np.count_nonzero(depth_8x8 > 0.0)
        cv2.putText(sidebar, f"Active: {active_cnt}/64 cells", (185, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(sidebar, "Wall: Null (0.0)", (185, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        if active_cnt > 0:
            active_min = depth_8x8[depth_8x8 > 0.0].min() * 100.0
            active_max = depth_8x8[depth_8x8 > 0.0].max() * 100.0
            cv2.putText(sidebar, f"Span: {active_min:.1f}-{active_max:.1f}cm", (185, 495),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 220, 120), 1)
        else:
            cv2.putText(sidebar, "Span: No Active Zones", (185, 495),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        # Controls Footer
        cv2.line(sidebar, (20, 580), (SIDEBAR_WIDTH - 20, 580), (60, 60, 60), 1)
        cv2.putText(sidebar, "T               : Toggle Disk .npy / Live Tune", (20, 597),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)
        cv2.putText(sidebar, "N / P (or [ / ]) : Jump Next/Prev Gesture", (20, 613),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (180, 180, 180), 1)
        cv2.putText(sidebar, "LEFT / RIGHT     : Prev/Next Image | R: Random | Q: Quit", (20, 629),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)

        # Quit: Q / ESC
        if key in (ord('q'), ord('Q'), 27):
            break

        # Toggle Disk vs. Live Synthesized: 'T'
        elif key in (ord('t'), ord('T')):
            USE_DISK_NPY = not USE_DISK_NPY

        # Up Arrow (Linux: 65362, Win: 2490368, 'W') -> Increase Distance (+0.2 cm)
        elif key in (65362, 2490368, 82, ord('w'), ord('W')):
            current_distance_cm = round(min(current_distance_cm + DISTANCE_STEP_CM, MAX_DISTANCE_CM), 2)

        # Down Arrow (Linux: 65364, Win: 2621440, 'S') -> Decrease Distance (-0.2 cm)
        elif key in (65364, 2621440, 84, ord('s'), ord('S')):
            current_distance_cm = round(max(current_distance_cm - DISTANCE_STEP_CM, MIN_DISTANCE_CM), 2)

        # Left Arrow (Linux: 65361, Win: 2424832, 'A') -> Previous Image
        elif key in (65361, 2424832, 81, ord('a'), ord('A')):
            current_idx = (current_idx - 1) % len(samples)

        # Right Arrow (Linux: 65363, Win: 2555904, 'D') -> Next Image
        elif key in (65363, 2555904, 83, ord('d'), ord('D')):
            current_idx = (current_idx + 1) % len(samples)

        # Next Gesture Folder: 'N' or ']'
        elif key in (ord('n'), ord('N'), ord(']')):
            current_idx = find_gesture_jump(samples, current_idx, direction=1)

        # Previous Gesture Folder: 'P' or '['
        elif key in (ord('p'), ord('P'), ord('[')):
            current_idx = find_gesture_jump(samples, current_idx, direction=-1)

        # Random Image: 'R'
        elif key in (ord('r'), ord('R')):
            current_idx = random.randint(0, len(samples) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()