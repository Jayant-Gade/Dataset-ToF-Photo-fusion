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
CAM_ID = 0                     # Web camera index (0 = default internal/external webcam)

BASE_DATASET_DIR = "gesturanet_augmented_dataset" if os.path.isdir("gesturanet_augmented_dataset") else "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"    # Retained folder name; calibrated to 30 cm
WEIGHTS_PATH = "weights/gesturanet_4class_best.pth"

ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign"
]

# Mode Flags
USE_DISK_NPY = True            # True = load .npy from disk; False = live synthetic ToF
BLACKOUT_RGB = False           # True = blackout camera pixels to test pure ToF branch
IS_LIVE_CAM_MODE = False       # True = streaming from webcam; False = dataset inspection
IS_CAM_FROZEN = False          # True = holds captured live frame for distance/blackout tweaks

# UI Dimensions
VIEWPORT_SIZE = 640
SIDEBAR_WIDTH = 420
WINDOW_NAME = "GesturaNet - Multimodal Live & Dataset Inspector"

# Sensor Simulation & Calibration Parameters
DEFAULT_DISTANCE_CM = 30.0     # 30.0 cm baseline calibration
MIN_DISTANCE_CM = 15.0         # Minimum tuning limit
MAX_DISTANCE_CM = 50.0         # Maximum tuning limit
DISTANCE_STEP_CM = 0.2         # Tuning increment per keypress
NULL_VALUE_M = 0.000           # Background / Open air
NOISE_PERCENT = 0.015          # 1.5% sensor noise (~4.5 mm at 30 cm)
ZONE_ACTIVATION_THRESHOLD = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class GatedMultimodalGesturaNet(nn.Module):
    """
    Balanced Two-Stream Architecture with Gated Multimodal Fusion (GMU):
    - RGB Stream: MobileNetV3-Small -> 128-d projection[cite: 1]
    - Depth Stream: Spatial Conv2D (preserves 8x8 geometry) -> 128-d projection[cite: 1, 2]
    - Fusion Gate: Dynamic attention balancing RGB vs ToF
    """
    def __init__(self, num_classes=4):
        super().__init__()

        # --- 1. RGB Stream (128-d) ---
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_rgb = mobilenet_v3_small(weights=weights)
        self.rgb_features = base_rgb.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.rgb_proj = nn.Sequential(
            nn.Linear(576, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        # --- 2. ToF Depth Stream (128-d Spatial Preserving) ---
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 64, 4, 4)[cite: 2]
        )

        self.depth_proj = nn.Sequential(
            nn.Flatten(),                          # 64 * 4 * 4 = 1024
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        # --- 3. Gated Fusion Unit ---
        self.gate_layer = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Sigmoid()
        )

        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

        self.aux_depth_classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, depth, return_aux=False):
        f_rgb = self.rgb_pool(self.rgb_features(rgb)).flatten(1)
        emb_rgb = self.rgb_proj(f_rgb)

        f_depth = self.depth_conv(depth)
        emb_depth = self.depth_proj(f_depth)

        concat_feats = torch.cat((emb_rgb, emb_depth), dim=1)
        gate = self.gate_layer(concat_feats)
        fused_emb = gate * emb_rgb + (1.0 - gate) * emb_depth

        logits = self.classifier(fused_emb)

        if return_aux:
            aux_logits = self.aux_depth_classifier(emb_depth)
            return logits, aux_logits, gate

        return logits, gate


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


def apply_distance_shift_to_disk_tof(disk_depth, current_dist_cm):
    """
    Applies real-time distance delta tuning to precomputed .npy files on disk.
    Shifts active hand zones proportionally while keeping null background at 0.0.
    """
    delta_m = (current_dist_cm - DEFAULT_DISTANCE_CM) / 100.0
    shifted_depth = disk_depth.copy()
    active_mask = shifted_depth > 0.0
    if np.any(active_mask):
        shifted_depth[active_mask] = np.clip(shifted_depth[active_mask] + delta_m, 0.10, 1.00)
    return shifted_depth


def center_crop_square(frame):
    """Crops rectangular webcam feeds into a 1:1 symmetric square."""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cv2.resize(cropped, (VIEWPORT_SIZE, VIEWPORT_SIZE))


def draw_8x8_overlay(image):
    overlay = image.copy()
    step = VIEWPORT_SIZE // 8
    for i in range(1, 8):
        pos = i * step
        cv2.line(overlay, (pos, 0), (pos, VIEWPORT_SIZE), (0, 255, 0), 1)
        cv2.line(overlay, (0, pos), (VIEWPORT_SIZE, pos), (0, 255, 0), 1)
    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


def create_heatmap_view(depth_matrix, size=140):
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
    dataset_records = []
    for g_idx, gesture in enumerate(ACTIVE_GESTURES):
        folder = os.path.join(BASE_DATASET_DIR, gesture, TARGET_DISTANCE_DIR)
        files = sorted(glob.glob(os.path.join(folder, "img_*.jpg")))
        for f in files:
            dataset_records.append({"path": f, "gesture": gesture, "g_idx": g_idx})
    return dataset_records


def find_gesture_jump(samples, current_idx, direction):
    total = len(samples)
    current_g_idx = samples[current_idx]["g_idx"]
    target_g_idx = (current_g_idx + direction) % len(ACTIVE_GESTURES)

    for i, item in enumerate(samples):
        if item["g_idx"] == target_g_idx:
            return i
    return (current_idx + direction) % total


def main():
    global USE_DISK_NPY, BLACKOUT_RGB, IS_LIVE_CAM_MODE, IS_CAM_FROZEN
    samples = load_dataset_samples()

    # Load Model with the upgraded Gated Architecture
    model = GatedMultimodalGesturaNet(num_classes=len(ACTIVE_GESTURES)).to(DEVICE)
    if os.path.isfile(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"Loaded trained Gated Multimodal weights from: {WEIGHTS_PATH}")
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

    # Live Webcam Objects
    cap = None
    last_captured_frame = None

    print("\n--- GesturaNet Validation Inspector ---")
    print(f"Camera Index: {CAM_ID}")
    print("Press [C] to toggle Live Camera vs. Dataset Mode.")
    print("---------------------------------------\n")

    while True:
        # =========================================================================
        # 1. Image Sourcing (Live Webcam vs. Dataset Mode)
        # =========================================================================
        if IS_LIVE_CAM_MODE:
            if cap is None:
                cap = cv2.VideoCapture(CAM_ID)
                if not cap.isOpened():
                    print(f"Error: Unable to open webcam on CAM_ID = {CAM_ID}. Falling back to dataset.")
                    IS_LIVE_CAM_MODE = False
                    cap = None
                    continue

            if not IS_CAM_FROZEN or last_captured_frame is None:
                ret, frame = cap.read()
                if not ret:
                    continue
                raw_bgr = center_crop_square(frame)
                last_captured_frame = raw_bgr.copy()
            else:
                raw_bgr = last_captured_frame.copy()

            ground_truth = "LIVE WEBCAM"
            g_num = 0
            cached_mask = extract_robust_hand_mask(raw_bgr)
            depth_8x8 = synthesize_tof_array(cached_mask, current_distance_cm)
            tof_source_label = "LIVE SYNTHESIS"

        else:
            # Dataset Mode
            if cap is not None:
                cap.release()
                cap = None

            if not samples:
                continue

            record = samples[current_idx]
            img_path = record["path"]
            ground_truth = record["gesture"]
            g_num = record["g_idx"] + 1

            if current_idx != last_loaded_idx:
                raw_bgr = cv2.imread(img_path)
                if raw_bgr.shape[0] != VIEWPORT_SIZE or raw_bgr.shape[1] != VIEWPORT_SIZE:
                    raw_bgr = cv2.resize(raw_bgr, (VIEWPORT_SIZE, VIEWPORT_SIZE))
                cached_mask = extract_robust_hand_mask(raw_bgr)
                last_loaded_idx = current_idx

            base_no_ext = os.path.splitext(img_path)[0]
            saved_tof_path = f"{base_no_ext}_tof.npy"

            # Disk Load with dynamic distance shift
            if USE_DISK_NPY and os.path.isfile(saved_tof_path):
                raw_loaded = np.load(saved_tof_path).astype(np.float32)
                raw_clean = np.nan_to_num(raw_loaded, nan=0.0, posinf=0.0, neginf=0.0)
                depth_8x8 = apply_distance_shift_to_disk_tof(raw_clean, current_distance_cm)

                shift_delta = current_distance_cm - DEFAULT_DISTANCE_CM
                if abs(shift_delta) > 0.05:
                    tof_source_label = f"DISK ({shift_delta:+.1f}cm)"
                else:
                    tof_source_label = "DISK (.npy)"
            else:
                depth_8x8 = synthesize_tof_array(cached_mask, current_distance_cm)
                tof_source_label = "SYNTHESIZED"

        # =========================================================================
        # 2. Multimodal Inference
        # =========================================================================
        pil_img = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
        rgb_tensor = rgb_transforms(pil_img).unsqueeze(0).to(DEVICE)

        if BLACKOUT_RGB:
            rgb_tensor = torch.zeros_like(rgb_tensor)

        depth_tensor = torch.from_numpy(depth_8x8).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, gate = model(rgb_tensor, depth_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            rgb_weight = float(gate.mean().item()) * 100.0
            tof_weight = 100.0 - rgb_weight

        results = [
            {"gesture": ACTIVE_GESTURES[i], "score": float(probs[i])}
            for i in range(len(ACTIVE_GESTURES))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        top_prediction = results[0]["gesture"]
        is_correct = (top_prediction == ground_truth) if not IS_LIVE_CAM_MODE else True

        # =========================================================================
        # 3. Viewport (Left)
        # =========================================================================
        if BLACKOUT_RGB:
            viewport = np.zeros((VIEWPORT_SIZE, VIEWPORT_SIZE, 3), dtype=np.uint8)
            cv2.putText(viewport, "RGB BLACKOUT (PURE ToF INFERENCE)", (60, VIEWPORT_SIZE // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(viewport, "Press [B] to restore camera view", (140, (VIEWPORT_SIZE // 2) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        else:
            viewport = draw_8x8_overlay(raw_bgr)

        if IS_LIVE_CAM_MODE:
            state_text = "[LIVE - FROZEN/CAPTURED]" if IS_CAM_FROZEN else "[LIVE - STREAMING]"
            state_color = (0, 165, 255) if IS_CAM_FROZEN else (0, 255, 0)
            cv2.putText(viewport, state_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.65, state_color, 2)
            cv2.putText(viewport, "Press [SPACE] to Freeze/Unfreeze Capture", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        canvas[:, :VIEWPORT_SIZE] = viewport

        # =========================================================================
        # 4. Sidebar (Right)
        # =========================================================================
        sidebar = canvas[:, VIEWPORT_SIZE:]
        sidebar[:] = (26, 28, 32)

        # Header Section
        cv2.putText(sidebar, "VALIDATION BENCH", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 42), (SIDEBAR_WIDTH - 20, 42), (65, 65, 65), 1)

        # Sample Info & Mode
        if IS_LIVE_CAM_MODE:
            cv2.putText(sidebar, f"Source: LIVE WEBCAM (CAM_ID: {CAM_ID})", (20, 68),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 200), 1)
            cv2.putText(sidebar, "Mode: Real-Time Stream", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        else:
            fname = os.path.basename(samples[current_idx]["path"])
            cv2.putText(sidebar, f"Folder [{g_num}/{len(ACTIVE_GESTURES)}]: {ground_truth.upper()}", (20, 68),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 200), 1)
            cv2.putText(sidebar, f"File [{current_idx + 1}/{len(samples)}]: {fname}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Depth Source Card
        cv2.rectangle(sidebar, (18, 104), (SIDEBAR_WIDTH - 18, 168), (38, 42, 48), -1)
        cv2.rectangle(sidebar, (18, 104), (SIDEBAR_WIDTH - 18, 168), (70, 75, 82), 1)

        source_color = (0, 255, 120) if "DISK" in tof_source_label else (0, 200, 255)
        cv2.putText(sidebar, f"ToF Source: {tof_source_label}", (28, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, source_color, 1)

        cv2.putText(sidebar, f"Depth Scale: {current_distance_cm:.1f} cm", (28, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2)
        cv2.putText(sidebar, "UP / DOWN: Tune Depth | [T]: Toggle Disk", (28, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

        # Prediction Confidence Card
        cv2.rectangle(sidebar, (18, 178), (SIDEBAR_WIDTH - 18, 365), (32, 35, 40), -1)
        cv2.rectangle(sidebar, (18, 178), (SIDEBAR_WIDTH - 18, 365), (60, 65, 72), 1)

        verdict_color = (0, 210, 0) if is_correct else (0, 60, 230)
        verdict_text = "PREDICTED GESTURE" if IS_LIVE_CAM_MODE else ("MATCH CORRECT" if is_correct else "MISMATCH")
        mode_tag = " [BLACKOUT]" if BLACKOUT_RGB else ""
        cv2.putText(sidebar, f"Top: {top_prediction.upper()}{mode_tag}", (28, 202),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, verdict_color, 1)
        cv2.putText(sidebar, f"[{verdict_text}]", (28, 222),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, verdict_color, 1)

        # Class Probability Bars
        bar_start_y = 246
        for i, item in enumerate(results):
            g_name = item["gesture"]
            score = item["score"]
            y_pos = bar_start_y + (i * 28)

            txt = f"{g_name:<11} {score * 100:5.1f}%"
            txt_color = (255, 255, 255) if i == 0 else (170, 170, 170)
            cv2.putText(sidebar, txt, (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_color, 1)

            bar_x = 185
            bar_w = 195
            cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + bar_w, y_pos - 2), (50, 52, 58), -1)

            fill_w = int(bar_w * score)
            fill_color = (0, 200, 120) if i == 0 else (120, 120, 120)
            if fill_w > 0:
                cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + fill_w, y_pos - 2), fill_color, -1)

        # Attention Gate Split HUD
        cv2.putText(sidebar, f"Gate: RGB {rgb_weight:.0f}% | ToF {tof_weight:.0f}%", (20, 385),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
        gate_bar_w = SIDEBAR_WIDTH - 40
        gate_fill_w = int(gate_bar_w * (rgb_weight / 100.0))
        cv2.rectangle(sidebar, (20, 392), (20 + gate_bar_w, 400), (80, 50, 150), -1)
        cv2.rectangle(sidebar, (20, 392), (20 + gate_fill_w, 400), (0, 200, 255), -1)

        # 8x8 ToF Heatmap Preview
        cv2.putText(sidebar, "8x8 ToF Depth Map (0.0 = Null)", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        heatmap_view = create_heatmap_view(depth_8x8, size=140)
        sidebar[432:432 + 140, 20:20 + 140] = heatmap_view

        active_cnt = np.count_nonzero(depth_8x8 > 0.0)
        cv2.putText(sidebar, f"Active: {active_cnt}/64 zones", (180, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(sidebar, "Wall: Null (0.0)", (180, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        if active_cnt > 0:
            active_min = depth_8x8[depth_8x8 > 0.0].min() * 100.0
            active_max = depth_8x8[depth_8x8 > 0.0].max() * 100.0
            cv2.putText(sidebar, f"Span: {active_min:.1f}-{active_max:.1f}cm", (180, 505),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 220, 120), 1)
        else:
            cv2.putText(sidebar, "Span: No Active Zones", (180, 505),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        # Controls Footer
        cv2.line(sidebar, (20, 582), (SIDEBAR_WIDTH - 20, 582), (60, 60, 60), 1)
        cv2.putText(sidebar, "C: Toggle Live Webcam / Dataset | SPACE: Freeze", (20, 598),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1)
        cv2.putText(sidebar, "B: Blackout RGB | UP/DN: Tune Depth | T: Toggle Disk", (20, 613),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        cv2.putText(sidebar, "N/P: Jump Folder | LEFT/RIGHT: Step | Q: Quit", (20, 629),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imshow(WINDOW_NAME, canvas)

        # Polling rate: 10ms in live streaming for responsive video, wait indefinitely on dataset
        wait_time = 10 if (IS_LIVE_CAM_MODE and not IS_CAM_FROZEN) else 0
        key = cv2.waitKeyEx(wait_time)

        # Quit: Q / ESC
        if key in (ord('q'), ord('Q'), 27):
            break

        # Toggle Live Webcam Mode: 'C'
        elif key in (ord('c'), ord('C')):
            IS_LIVE_CAM_MODE = not IS_LIVE_CAM_MODE
            IS_CAM_FROZEN = False

        # Freeze/Capture Frame in Live Mode: 'SPACE'
        elif key == 32:  # ASCII for Spacebar
            if IS_LIVE_CAM_MODE:
                IS_CAM_FROZEN = not IS_CAM_FROZEN

        # Blackout RGB Toggle: 'B'
        elif key in (ord('b'), ord('B')):
            BLACKOUT_RGB = not BLACKOUT_RGB

        # Toggle Disk vs. Live Synthesized: 'T'
        elif key in (ord('t'), ord('T')):
            USE_DISK_NPY = not USE_DISK_NPY

        # Up Arrow: Increase Distance (+0.2 cm)
        elif key in (65362, 2490368, 82, ord('w'), ord('W')):
            current_distance_cm = round(min(current_distance_cm + DISTANCE_STEP_CM, MAX_DISTANCE_CM), 2)

        # Down Arrow: Decrease Distance (-0.2 cm)
        elif key in (65364, 2621440, 84, ord('s'), ord('S')):
            current_distance_cm = round(max(current_distance_cm - DISTANCE_STEP_CM, MIN_DISTANCE_CM), 2)

        # Dataset Navigation (Only when not in live mode)
        if not IS_LIVE_CAM_MODE and samples:
            if key in (65361, 2424832, 81, ord('a'), ord('A')):
                current_idx = (current_idx - 1) % len(samples)
            elif key in (65363, 2555904, 83, ord('d'), ord('D')):
                current_idx = (current_idx + 1) % len(samples)
            elif key in (ord('n'), ord('N'), ord(']')):
                current_idx = find_gesture_jump(samples, current_idx, direction=1)
            elif key in (ord('p'), ord('P'), ord('[')):
                current_idx = find_gesture_jump(samples, current_idx, direction=-1)
            elif key in (ord('r'), ord('R')):
                current_idx = random.randint(0, len(samples) - 1)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()