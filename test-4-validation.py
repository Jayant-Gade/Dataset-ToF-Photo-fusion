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
BASE_DATASET_DIR = "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"
WEIGHTS_PATH = "weights/gesturanet_4class_best.pth"

ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign"
]

# UI Dimensions
VIEWPORT_SIZE = 640
SIDEBAR_WIDTH = 380
WINDOW_NAME = "GesturaNet - Validation & Depth Sensitivity Inspector"

# Sensor Simulation Parameters
DEFAULT_DISTANCE_CM = 3.0
WALL_DISTANCE_M = 1.20
NOISE_PERCENT = 0.015
MIN_RANGE_M = 0.02
MAX_RANGE_M = 4.00

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class TwoStreamGesturaNet(nn.Module):
    """Multimodal fusion model matching the trained architecture."""
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


def segment_hand_mask(image_bgr):
    """Generates binary mask of hand against the white wall."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    l_channel = lab[:, :, 0]

    _, sat_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
    _, l_mask = cv2.threshold(l_channel, 215, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(sat_mask, l_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(cleaned)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 800:
            cv2.drawContours(hand_mask, [largest], -1, 255, -1)
    return hand_mask


def synthesize_tof_array(hand_mask, dist_cm):
    """Calculates 8x8 depth array using the user-adjusted distance and sensor noise[cite: 1]."""
    hand_dist_m = dist_cm / 100.0
    h, w = hand_mask.shape
    cell_h, cell_w = h // 8, w // 8
    depth_matrix = np.full((8, 8), WALL_DISTANCE_M, dtype=np.float32)

    for r in range(8):
        for c in range(8):
            patch = hand_mask[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            coverage = np.count_nonzero(patch) / (cell_h * cell_w)

            if coverage >= 0.15:
                base_val = hand_dist_m
            elif coverage > 0.03:
                t = (coverage - 0.03) / 0.12
                base_val = (1.0 - t) * WALL_DISTANCE_M + t * hand_dist_m
            else:
                base_val = WALL_DISTANCE_M

            noise = np.random.normal(0.0, base_val * NOISE_PERCENT)
            depth_matrix[r, c] = np.clip(base_val + noise, MIN_RANGE_M, MAX_RANGE_M)

    return depth_matrix


def draw_8x8_overlay(image):
    overlay = image.copy()
    step = VIEWPORT_SIZE // 8
    for i in range(1, 8):
        pos = i * step
        cv2.line(overlay, (pos, 0), (pos, VIEWPORT_SIZE), (0, 255, 0), 1)
        cv2.line(overlay, (0, pos), (VIEWPORT_SIZE, pos), (0, 255, 0), 1)
    return cv2.addWeighted(overlay, 0.35, image, 0.65, 0)


def create_heatmap_view(depth_matrix, size=160):
    """Renders a colorized mini-heatmap for the sidebar."""
    norm = np.clip((depth_matrix - 0.02) / (WALL_DISTANCE_M - 0.02), 0.0, 1.0)
    heat = (255 * (1.0 - norm)).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    return cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)


def load_dataset_samples():
    dataset_records = []
    for gesture in ACTIVE_GESTURES:
        folder = os.path.join(BASE_DATASET_DIR, gesture, TARGET_DISTANCE_DIR)
        files = sorted(glob.glob(os.path.join(folder, "img_*.jpg")))
        for f in files:
            dataset_records.append({"path": f, "gesture": gesture})
    return dataset_records


def main():
    samples = load_dataset_samples()
    if not samples:
        print(f"No captured images found in {BASE_DATASET_DIR} across classes: {ACTIVE_GESTURES}")
        return

    print(f"Loaded {len(samples)} images for validation inspection.")

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

        # Cache mask computation unless image index changes
        if current_idx != last_loaded_idx:
            raw_bgr = cv2.imread(img_path)
            if raw_bgr.shape[0] != VIEWPORT_SIZE or raw_bgr.shape[1] != VIEWPORT_SIZE:
                raw_bgr = cv2.resize(raw_bgr, (VIEWPORT_SIZE, VIEWPORT_SIZE))
            cached_mask = segment_hand_mask(raw_bgr)
            last_loaded_idx = current_idx

        # 1. Simulate ToF Depth array at current_distance_cm
        depth_8x8 = synthesize_tof_array(cached_mask, current_distance_cm)

        # 2. Run Model Inference
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

        # 3. Render Viewport (Left)
        viewport = draw_8x8_overlay(raw_bgr)
        canvas[:, :VIEWPORT_SIZE] = viewport

        # 4. Render Dashboard Sidebar (Right)
        sidebar = canvas[:, VIEWPORT_SIZE:]
        sidebar[:] = (26, 28, 32)  # Charcoal background

        # Header Section
        cv2.putText(sidebar, "VALIDATION BENCH", (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        cv2.line(sidebar, (20, 42), (SIDEBAR_WIDTH - 20, 42), (65, 65, 65), 1)

        # Sample Info
        fname = os.path.basename(img_path)
        cv2.putText(sidebar, f"Sample: [{current_idx + 1}/{len(samples)}] {fname}", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)
        cv2.putText(sidebar, f"Ground Truth: {ground_truth}", (20, 92),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1)

        # Distance Dial Section
        cv2.rectangle(sidebar, (18, 108), (SIDEBAR_WIDTH - 18, 168), (38, 42, 48), -1)
        cv2.rectangle(sidebar, (18, 108), (SIDEBAR_WIDTH - 18, 168), (70, 75, 82), 1)

        cv2.putText(sidebar, f"Simulated Distance: {current_distance_cm:.2f} cm", (28, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(sidebar, "Adjust: UP (+0.1cm) / DOWN (-0.1cm)", (28, 154),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        # Prediction Confidence Card
        cv2.rectangle(sidebar, (18, 182), (SIDEBAR_WIDTH - 18, 385), (32, 35, 40), -1)
        cv2.rectangle(sidebar, (18, 182), (SIDEBAR_WIDTH - 18, 385), (60, 65, 72), 1)

        # Verdict Badge
        verdict_color = (0, 210, 0) if is_correct else (0, 60, 230)
        verdict_text = "MATCH CORRECT" if is_correct else "MISMATCH"
        cv2.putText(sidebar, f"Top: {top_prediction.upper()}", (28, 208),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, verdict_color, 1)
        cv2.putText(sidebar, f"[{verdict_text}]", (28, 228),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, verdict_color, 1)

        # Sorted Predictions & Progress Bars
        bar_start_y = 250
        for i, item in enumerate(results):
            g_name = item["gesture"]
            score = item["score"]
            y_pos = bar_start_y + (i * 32)

            # Class label & percentage
            txt = f"{g_name:<11} {score * 100:5.1f}%"
            txt_color = (255, 255, 255) if i == 0 else (170, 170, 170)
            cv2.putText(sidebar, txt, (28, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, 1)

            # Bar track
            bar_x = 180
            bar_w = 160
            cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + bar_w, y_pos - 2), (50, 52, 58), -1)

            # Fill bar
            fill_w = int(bar_w * score)
            fill_color = (0, 200, 120) if i == 0 else (130, 130, 130)
            if fill_w > 0:
                cv2.rectangle(sidebar, (bar_x, y_pos - 10), (bar_x + fill_w, y_pos - 2), fill_color, -1)

        # Heatmap & Sensor Preview
        cv2.putText(sidebar, "8x8 ToF Heatmap", (20, 412), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        heatmap_view = create_heatmap_view(depth_8x8, size=140)
        sidebar[425:425 + 140, 20:20 + 140] = heatmap_view

        # Hot/Cold distance legend text
        cv2.putText(sidebar, f"Hand: {current_distance_cm:.1f} cm", (175, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(sidebar, f"Wall: {WALL_DISTANCE_M*100:.0f} cm", (175, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
        cv2.putText(sidebar, f"Noise: +-{current_distance_cm*NOISE_PERCENT*10:.2f}mm", (175, 505),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        # Controls Footer
        cv2.line(sidebar, (20, 580), (SIDEBAR_WIDTH - 20, 580), (60, 60, 60), 1)
        cv2.putText(sidebar, "LEFT / RIGHT : Prev/Next Image | R: Random", (20, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        cv2.putText(sidebar, "UP / DOWN    : Adjust Distance (+-0.1 cm)", (20, 618),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
        cv2.putText(sidebar, "Q / ESC      : Quit Validation", (20, 634),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)

        # Quit
        if key in (ord('q'), ord('Q'), 27):
            break

        # Up Arrow (Linux: 65362, Win: 2490368, Fallback: 'w') -> Increase Distance (+0.1 cm)
        elif key in (65362, 2490368, 82, ord('w'), ord('W')):
            current_distance_cm = round(min(current_distance_cm + 0.1, 15.0), 2)

        # Down Arrow (Linux: 65364, Win: 2621440, Fallback: 's') -> Decrease Distance (-0.1 cm)
        elif key in (65364, 2621440, 84, ord('s'), ord('S')):
            current_distance_cm = round(max(current_distance_cm - 0.1, 1.0), 2)

        # Left Arrow (Linux: 65361, Win: 2424832, Fallback: 'a') -> Previous Image
        elif key in (65361, 2424832, 81, ord('a'), ord('A')):
            current_idx = (current_idx - 1) % len(samples)

        # Right Arrow (Linux: 65363, Win: 2555904, Fallback: 'd') -> Next Image
        elif key in (65363, 2555904, 83, ord('d'), ord('D')):
            current_idx = (current_idx + 1) % len(samples)

        # 'R' Key -> Pick Random Image
        elif key in (ord('r'), ord('R')):
            current_idx = random.randint(0, len(samples) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()