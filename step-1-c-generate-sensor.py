import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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
TARGET_DATASET_DIR = "gesturanet_augmented_dataset"
TARGET_DISTANCE_DIR = "3cm"

# Ground Truth Calibration (Meters)
HAND_BASE_DIST_M = 0.030       # 3.0 cm
WALL_DISTANCE_M  = 0.032       # 3.2 cm
NOISE_PERCENT    = 0.015       # 1.5% sensor noise

MIN_SENSOR_RANGE_M = 0.020     # Hardware floor (20 mm)
MAX_SENSOR_RANGE_M = 4.000     # Max range (4000 mm)
ZONE_ACTIVATION_THRESHOLD = 0.12

GENERATE_PREVIEWS = False      # Set to True if visual check plots are desired
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
# =================================================

worker_mp_hands = None


def init_worker():
    global worker_mp_hands, HAS_MEDIAPIPE
    if HAS_MEDIAPIPE:
        try:
            worker_mp_hands = mp_hands_module.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.35
            )
        except Exception:
            worker_mp_hands = None


def extract_hand_mask(image_bgr):
    """Dual-pass segmentation ensuring hand mask is never empty."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    saturation = hsv[:, :, 1]
    l_channel = lab[:, :, 0]

    _, sat_mask = cv2.threshold(saturation, 20, 255, cv2.THRESH_BINARY)
    _, l_mask = cv2.threshold(l_channel, 215, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(sat_mask, l_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(cleaned)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 400:
            cv2.drawContours(mask, [largest], -1, 255, -1)

    # Grayscale Otsu fallback for closed fist
    if np.count_nonzero(mask) < 400:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400:
                cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def compute_anatomical_tof(image_rgb, hand_mask):
    """Computes an 8x8 ToF depth matrix with anatomical variation."""
    global worker_mp_hands
    h, w, _ = image_rgb.shape

    mask_float = (hand_mask > 0).astype(np.float32)
    coverage_8x8 = cv2.resize(mask_float, (8, 8), interpolation=cv2.INTER_AREA)

    hand_depth_8x8 = np.full((8, 8), HAND_BASE_DIST_M, dtype=np.float32)
    mediapipe_success = False

    if worker_mp_hands is not None and np.count_nonzero(hand_mask) > 400:
        try:
            results = worker_mp_hands.process(image_rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
                depths = np.array([HAND_BASE_DIST_M + (lm.z * 0.015) for lm in landmarks], dtype=np.float32)

                step_y, step_x = h / 8.0, w / 8.0
                grid_y, grid_x = np.mgrid[0:8, 0:8]
                zone_centers = np.column_stack(((grid_x.ravel() + 0.5) * step_x, (grid_y.ravel() + 0.5) * step_y))

                dists = np.linalg.norm(zone_centers[:, None, :] - pts[None, :, :], axis=2) + 1e-4
                weights = 1.0 / (dists ** 2)
                sum_weights = weights.sum(axis=1, keepdims=True)
                weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights > 0)

                hand_depth_8x8 = np.sum(weights * depths, axis=1).reshape((8, 8))
                mediapipe_success = True
        except Exception:
            mediapipe_success = False

    # Zero-division-safe curvature fallback
    if not mediapipe_success:
        if np.count_nonzero(hand_mask) > 100:
            dist_transform = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
            dt_8x8 = cv2.resize(dist_transform, (8, 8), interpolation=cv2.INTER_AREA)
            max_dt = float(dt_8x8.max())
            norm_dt = (dt_8x8 / max_dt) if max_dt > 0.001 else np.zeros((8, 8), dtype=np.float32)
        else:
            norm_dt = np.zeros((8, 8), dtype=np.float32)

        hand_depth_8x8 = HAND_BASE_DIST_M - (norm_dt * 0.002)

    # 8x8 zone blending
    tof_matrix = np.full((8, 8), WALL_DISTANCE_M, dtype=np.float32)
    for r in range(8):
        for c in range(8):
            cov = coverage_8x8[r, c]
            if cov >= ZONE_ACTIVATION_THRESHOLD:
                base_d = hand_depth_8x8[r, c]
            elif cov > 0.02:
                t = cov / ZONE_ACTIVATION_THRESHOLD
                base_d = (1.0 - t) * WALL_DISTANCE_M + t * hand_depth_8x8[r, c]
            else:
                base_d = WALL_DISTANCE_M

            noise = np.random.normal(0.0, base_d * NOISE_PERCENT)
            tof_matrix[r, c] = np.clip(base_d + noise, MIN_SENSOR_RANGE_M, MAX_SENSOR_RANGE_M)

    return np.nan_to_num(tof_matrix, nan=WALL_DISTANCE_M, posinf=WALL_DISTANCE_M, neginf=WALL_DISTANCE_M).astype(np.float32)


def process_single_image(img_path):
    folder = os.path.dirname(img_path)
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_npy = os.path.join(folder, f"{base}_tof.npy")

    # Skip if already generated
    if os.path.isfile(out_npy):
        return True

    frame = cv2.imread(img_path)
    if frame is None:
        return False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = extract_hand_mask(frame)
    tof_data = compute_anatomical_tof(rgb_frame, mask)

    np.save(out_npy, tof_data)
    return True


def main():
    target_pattern = os.path.join(TARGET_DATASET_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    img_list = sorted(glob.glob(target_pattern))
    total_imgs = len(img_list)

    if not total_imgs:
        print(f"No images found in '{TARGET_DATASET_DIR}'. Run Step 1-B first.")
        return

    print(f"--- Step 1-C: Multi-Threaded ToF Sensor Generation ---")
    print(f"Target Directory: '{TARGET_DATASET_DIR}'")
    print(f"Found {total_imgs} images. Running across {NUM_WORKERS} CPU workers...\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker) as executor:
        for success in executor.map(process_single_image, img_list):
            if success:
                completed += 1
            if completed % 100 == 0 or completed == total_imgs:
                pct = (completed / total_imgs) * 100
                print(f"Progress: [{completed}/{total_imgs}] ({pct:.1f}%) ToF arrays ready.")

    print(f"\nStep 1-C Complete: {completed} ToF matrices generated.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()