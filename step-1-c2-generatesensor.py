import os
import glob
import cv2
import numpy as np

# Protobuf environment patch for Linux
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        mp_hands_module = mp.solutions.hands
        HAS_MEDIAPIPE = True
    else:
        from mediapipe.python.solutions import hands as mp_hands_module
        HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_augmented_dataset"
TARGET_DISTANCE_DIR = "3cm"   # Kept folder name for compatibility; values calibrated to 30 cm

# Physical Calibration (in meters)
HAND_BASE_DIST_M = 0.300       # 30.0 cm baseline distance
NULL_VALUE_M     = 0.000       # Null / out of bounds / open air
Z_SPAN_M         = 0.020       # Anatomical z-depth range (~29.0 cm to 32.0 cm)
NOISE_PERCENT    = 0.015       # 1.5% sensor noise (~4.5 mm at 30 cm)

MIN_SENSOR_RANGE_M = 0.020     # Hardware floor (20 mm)
MAX_SENSOR_RANGE_M = 4.000     # Hardware ceiling (4000 mm)
ZONE_ACTIVATION_THRESHOLD = 0.10

WINDOW_NAME = "GesturaNet - Step 1-C Interactive Studio (30cm / Null BG)"
PANEL_HEIGHT = 440
CARD_SIZE = 360
# =================================================

# Initialize MediaPipe detector
mp_detector = None
if HAS_MEDIAPIPE:
    try:
        mp_detector = mp_hands_module.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3
        )
    except Exception:
        mp_detector = None


def extract_robust_hand_mask(image_bgr):
    """
    Extracts skin using YCrCb Chromaticity (Cr - Cb) and HSV saturation.
    Cleanly separates hand and arm from off-white walls and background shadows.
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

    # Fallback to inverted Otsu if color threshold misses
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


def compute_tof_matrix(image_rgb, hand_mask):
    """
    Computes an 8x8 ToF depth matrix at 30 cm scale:
    - Hand zones receive 29.0 cm to 32.0 cm readings with 1.5% noise.
    - Background/empty zones receive 0.0 (Null).
    """
    h, w, _ = image_rgb.shape

    mask_float = (hand_mask > 0).astype(np.float32)
    coverage_8x8 = cv2.resize(mask_float, (8, 8), interpolation=cv2.INTER_AREA)

    hand_depth_8x8 = np.full((8, 8), HAND_BASE_DIST_M, dtype=np.float32)
    detection_method = "Fallback: Curvature"

    hand_pixels = np.count_nonzero(hand_mask)

    # 1. Attempt MediaPipe 3D joint landmarking
    if mp_detector is not None and hand_pixels > 400:
        try:
            results = mp_detector.process(image_rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

                # lm.z is relative to wrist; scale across the 29 cm - 32 cm span
                depths = np.array([HAND_BASE_DIST_M + (lm.z * Z_SPAN_M) for lm in landmarks], dtype=np.float32)

                step_y, step_x = h / 8.0, w / 8.0
                grid_y, grid_x = np.mgrid[0:8, 0:8]
                zone_centers = np.column_stack(((grid_x.ravel() + 0.5) * step_x, (grid_y.ravel() + 0.5) * step_y))

                dists = np.linalg.norm(zone_centers[:, None, :] - pts[None, :, :], axis=2) + 1e-4
                weights = 1.0 / (dists ** 2)
                sum_weights = weights.sum(axis=1, keepdims=True)
                weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights > 0)

                hand_depth_8x8 = np.sum(weights * depths, axis=1).reshape((8, 8))
                detection_method = "MediaPipe 3D Joints"
        except Exception:
            detection_method = "Fallback: Curvature"

    # 2. Curvature fallback for closed fists
    if detection_method.startswith("Fallback"):
        if hand_pixels > 200:
            dist_transform = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
            dt_8x8 = cv2.resize(dist_transform, (8, 8), interpolation=cv2.INTER_AREA)
            max_dt = float(dt_8x8.max())
            norm_dt = (dt_8x8 / max_dt) if max_dt > 0.001 else np.zeros((8, 8), dtype=np.float32)

            # Knuckles/fist core project closer (~29.0 cm), perimeter at ~31.0 cm
            hand_depth_8x8 = (HAND_BASE_DIST_M + 0.010) - (norm_dt * 0.020)
        else:
            hand_depth_8x8 = np.full((8, 8), HAND_BASE_DIST_M, dtype=np.float32)

    # 3. Zone masking: Hand = distance + noise; Background = NULL_VALUE_M (0.0)
    tof_matrix = np.full((8, 8), NULL_VALUE_M, dtype=np.float32)
    for r in range(8):
        for c in range(8):
            cov = coverage_8x8[r, c]
            if cov >= ZONE_ACTIVATION_THRESHOLD:
                base_d = hand_depth_8x8[r, c]
                noise = np.random.normal(0.0, base_d * NOISE_PERCENT)
                tof_matrix[r, c] = np.clip(base_d + noise, 0.25, 0.40)
            else:
                tof_matrix[r, c] = NULL_VALUE_M

    tof_matrix = np.nan_to_num(tof_matrix, nan=NULL_VALUE_M).astype(np.float32)
    return tof_matrix, detection_method, coverage_8x8


def render_tof_heatmap_view(tof_matrix, size=CARD_SIZE):
    """
    Renders 8x8 heatmap:
    - Active hand cells display in color with distances in cm (e.g. 29.5).
    - Null background cells display in dark gray with '---'.
    """
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)  # Dark background for nulls
    step = size // 8

    # Normalize active zones between 28.0 cm and 33.0 cm for colormap
    active_mask = tof_matrix > 0.0
    if np.any(active_mask):
        norm = np.clip((tof_matrix - 0.28) / (0.33 - 0.28), 0.0, 1.0)
        heat = (255 * (1.0 - norm)).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
        colored_resized = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)

        # Copy only active hand blocks
        for r in range(8):
            for c in range(8):
                if tof_matrix[r, c] > 0.0:
                    canvas[r * step:(r + 1) * step, c * step:(c + 1) * step] = \
                        colored_resized[r * step:(r + 1) * step, c * step:(c + 1) * step]

    # Draw cell text and grid borders
    for r in range(8):
        for c in range(8):
            val_m = tof_matrix[r, c]
            tx = c * step + int(step * 0.12)
            ty = r * step + int(step * 0.65)

            if val_m > 0.0:
                val_cm = val_m * 100.0
                txt = f"{val_cm:.1f}"
                txt_color = (0, 0, 0) if val_cm < 30.5 else (255, 255, 255)
            else:
                txt = "---"
                txt_color = (100, 100, 100)

            cv2.putText(canvas, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_color, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (c * step, r * step), ((c + 1) * step, (r + 1) * step), (50, 50, 50), 1)

    return canvas


def draw_8x8_overlay(image_bgr):
    overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    step_x = w // 8
    step_y = h // 8
    for i in range(1, 8):
        cv2.line(overlay, (i * step_x, 0), (i * step_x, h), (0, 255, 0), 1)
        cv2.line(overlay, (0, i * step_y), (w, i * step_y), (0, 255, 0), 1)
    return cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0)


def run_batch_processing(image_paths):
    total = len(image_paths)
    print(f"\n==========================================")
    print(f"STARTING BATCH RE-GENERATION FOR {total} IMAGES")
    print(f"Calibration: 30cm scale (29-32cm) | Background: NULL (0.0)")
    print(f"==========================================\n")

    success_count = 0
    for idx, p in enumerate(image_paths, 1):
        frame = cv2.imread(p)
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = extract_robust_hand_mask(frame)
        tof_data, _, _ = compute_tof_matrix(rgb_frame, mask)

        folder = os.path.dirname(p)
        base = os.path.splitext(os.path.basename(p))[0]
        out_npy = os.path.join(folder, f"{base}_tof.npy")

        np.save(out_npy, tof_data)
        success_count += 1

        if idx % 50 == 0 or idx == total:
            pct = (idx / total) * 100
            print(f"Progress: [{idx}/{total}] ({pct:.1f}%) -> Generated 30cm .npy")

    print(f"\nBATCH FINISHED: Successfully generated {success_count} clean .npy files.")
    print("Press any key in the Studio window to continue inspecting, or Q to exit.")


def main():
    pattern = os.path.join(BASE_DATASET_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    all_images = sorted(glob.glob(pattern))

    preview_images = [p for p in all_images if "_orig" in p or "_rot" in p]
    if not preview_images:
        preview_images = all_images

    if not preview_images:
        print(f"No images found in {BASE_DATASET_DIR}.")
        return

    current_idx = 0
    total_samples = len(preview_images)

    TOTAL_WIDTH = CARD_SIZE * 3 + 40
    canvas = np.zeros((PANEL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)

    print("\n--- GesturaNet Step 1-C Visual Inspector (30cm / Null BG) ---")
    print(f"Found {total_samples} images to preview.")
    print("Controls:")
    print("  [LEFT] / [RIGHT] (or A / D): Browse next/prev image")
    print("  [R]                        : Pick random image")
    print("  [ENTER]                    : RUN BATCH (Process and overwrite ALL images)")
    print("  [Q] / [ESC]                : Quit")
    print("------------------------------------------------------------\n")

    while True:
        img_path = preview_images[current_idx]
        frame = cv2.imread(img_path)
        if frame is None:
            current_idx = (current_idx + 1) % total_samples
            continue

        frame_resized = cv2.resize(frame, (CARD_SIZE, CARD_SIZE))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # 1. Hand Mask
        mask = extract_robust_hand_mask(frame_resized)
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 2. ToF depth array (30 cm scale with null background)
        tof_data, method_name, cov_grid = compute_tof_matrix(rgb_frame, mask)
        heatmap_vis = render_tof_heatmap_view(tof_data, size=CARD_SIZE)

        # 3. Assemble display cards
        rgb_with_grid = draw_8x8_overlay(frame_resized)

        canvas[:] = (26, 28, 32)
        canvas[10:10 + CARD_SIZE, 10:10 + CARD_SIZE] = rgb_with_grid
        canvas[10:10 + CARD_SIZE, 20 + CARD_SIZE:20 + CARD_SIZE * 2] = mask_vis
        canvas[10:10 + CARD_SIZE, 30 + CARD_SIZE * 2:30 + CARD_SIZE * 3] = heatmap_vis

        # Titles
        cv2.putText(canvas, "1. Input RGB (8x8 FoV)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(canvas, "2. Segmented Hand Mask", (25 + CARD_SIZE, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(canvas, "3. Synthetic ToF (29-32cm | 0=Null)", (35 + CARD_SIZE * 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        # Diagnostics footer
        active_cells = np.count_nonzero(tof_data > 0.0)
        min_cm = (tof_data[tof_data > 0.0].min() * 100.0) if active_cells > 0 else 0.0
        max_cm = (tof_data[tof_data > 0.0].max() * 100.0) if active_cells > 0 else 0.0
        fname = os.path.basename(img_path)
        g_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

        hud_y = PANEL_HEIGHT - 35
        cv2.putText(canvas, f"[{current_idx + 1}/{total_samples}] {g_name}/{fname}", (15, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(canvas, f"Method: {method_name} | Active cells: {active_cells}/64 | Range: {min_cm:.1f}cm - {max_cm:.1f}cm (Null: 0.0)",
                    (15, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 200), 1)

        # Action Prompt
        cv2.putText(canvas, "[ENTER]: RUN BATCH ON ALL IMAGES", (TOTAL_WIDTH - 350, hud_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(canvas, "A/D: Browse | R: Random | Q: Quit", (TOTAL_WIDTH - 350, hud_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKeyEx(0)

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (65363, 2555904, 83, ord('d'), ord('D')):
            current_idx = (current_idx + 1) % total_samples
        elif key in (65361, 2424832, 81, ord('a'), ord('A')):
            current_idx = (current_idx - 1) % total_samples
        elif key in (ord('r'), ord('R')):
            current_idx = np.random.randint(0, total_samples)
        elif key in (10, 13):
            run_batch_processing(all_images)


if __name__ == "__main__":
    main()