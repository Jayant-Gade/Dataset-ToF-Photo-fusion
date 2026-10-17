import os
import glob
import random
import shutil
import cv2
import numpy as np

# ================= CONFIGURATION =================
SRC_RAW_DIR = "gesturanet_raw_dataset"
DST_AUG_DIR = "gesturanet_augmented_dataset"
TARGET_DISTANCE_DIR = "3cm"

# Augmentation Multipliers
NUM_ROTATIONS_PER_IMAGE = 2    # Number of new geometric poses per original frame (+-8 deg)
PHOTOMETRIC_VARIANTS = 2       # Lighting/blur variations per pose (including original pose)

# Physical Sensor Baseline (Meters)
HAND_BASE_DIST_M = 0.030       # 3.0 cm
WALL_DISTANCE_M  = 0.032       # 3.2 cm
NOISE_PERCENT    = 0.015       # 1.5% Gaussian sensor jitter
MIN_SENSOR_RANGE_M = 0.020
MAX_SENSOR_RANGE_M = 4.000
ZONE_ACTIVATION_THRESHOLD = 0.12
# =================================================


def extract_hand_mask(image_bgr):
    """Extracts binary hand mask against the white wall."""
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

    if np.count_nonzero(mask) < 400:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 400:
                cv2.drawContours(mask, [largest], -1, 255, -1)

    return mask


def compute_tof_from_mask(mask_640):
    """
    Downsamples the 640x640 mask directly into the 8x8 VL53L8CX grid with
    zero-safe distance transforms and 1.5% sensor noise.
    """
    mask_float = (mask_640 > 0).astype(np.float32)
    coverage_8x8 = cv2.resize(mask_float, (8, 8), interpolation=cv2.INTER_AREA)

    if np.count_nonzero(mask_640) > 100:
        dist_transform = cv2.distanceTransform(mask_640, cv2.DIST_L2, 5)
        dt_8x8 = cv2.resize(dist_transform, (8, 8), interpolation=cv2.INTER_AREA)
        max_dt = float(dt_8x8.max())
        norm_dt = (dt_8x8 / max_dt) if max_dt > 0.001 else np.zeros((8, 8), dtype=np.float32)
    else:
        norm_dt = np.zeros((8, 8), dtype=np.float32)

    hand_depth_8x8 = HAND_BASE_DIST_M - (norm_dt * 0.002)

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

    return np.nan_to_num(tof_matrix, nan=WALL_DISTANCE_M).astype(np.float32)


def apply_high_res_geometric_transform(image_rgb, mask_640):
    """Rotates both RGB and hand mask simultaneously at 640x640 resolution."""
    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.96, 1.04)

    h, w = image_rgb.shape[:2]
    max_dx = int(w * 0.04)
    max_dy = int(h * 0.04)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    rot_rgb = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rot_mask = cv2.warpAffine(mask_640, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rot_rgb, rot_mask


def generate_photometric_variant(image_bgr, variant_idx):
    """Generates lighting/sensor variations (depth remains unchanged)."""
    img = image_bgr.astype(np.float32)

    if variant_idx == 1:
        # Darker / Underexposed
        alpha = random.uniform(0.70, 0.85)
        beta = random.uniform(-25.0, -10.0)
        img = img * alpha + beta
    elif variant_idx == 2:
        # Brighter / Overexposed
        alpha = random.uniform(1.15, 1.30)
        beta = random.uniform(10.0, 25.0)
        img = img * alpha + beta
    else:
        # Subtle contrast shift + minor defocus blur
        alpha = random.uniform(0.90, 1.10)
        img = img * alpha
        if random.random() > 0.4:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    target_pattern = os.path.join(SRC_RAW_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    raw_images = sorted(glob.glob(target_pattern))

    if not raw_images:
        print(f"No source images found in {SRC_RAW_DIR}")
        return

    print(f"Found {len(raw_images)} base captures in '{SRC_RAW_DIR}'.")
    print(f"Target augmented directory: '{DST_AUG_DIR}'")
    print(f"Generating {NUM_ROTATIONS_PER_IMAGE} rotation poses + {PHOTOMETRIC_VARIANTS} lighting variants each...\n")

    total_saved = 0

    for idx, img_path in enumerate(raw_images, 1):
        rel_path = os.path.relpath(img_path, SRC_RAW_DIR)
        sub_folder = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        tof_path = os.path.join(SRC_RAW_DIR, sub_folder, f"{base_name}_tof.npy")

        if not os.path.isfile(tof_path):
            continue

        target_sub_dir = os.path.join(DST_AUG_DIR, sub_folder)
        os.makedirs(target_sub_dir, exist_ok=True)

        raw_rgb = cv2.imread(img_path)
        raw_tof = np.load(tof_path).astype(np.float32)
        raw_mask = extract_hand_mask(raw_rgb)

        # 1. Save original pair
        orig_img_name = f"{base_name}_orig.jpg"
        orig_tof_name = f"{base_name}_orig_tof.npy"
        cv2.imwrite(os.path.join(target_sub_dir, orig_img_name), raw_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        np.save(os.path.join(target_sub_dir, orig_tof_name), raw_tof)
        total_saved += 1

        # 2. Save photometric variants of the original pose (reusing raw_tof)
        for p_idx in range(1, PHOTOMETRIC_VARIANTS + 1):
            p_img = generate_photometric_variant(raw_rgb, p_idx)
            p_name = f"{base_name}_orig_p{p_idx}.jpg"
            p_tof_name = f"{base_name}_orig_p{p_idx}_tof.npy"

            cv2.imwrite(os.path.join(target_sub_dir, p_name), p_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            np.save(os.path.join(target_sub_dir, p_tof_name), raw_tof)
            total_saved += 1

        # 3. Generate rotated poses + calculate exact rotated ToF matrices
        for r_idx in range(1, NUM_ROTATIONS_PER_IMAGE + 1):
            rot_rgb, rot_mask = apply_high_res_geometric_transform(raw_rgb, raw_mask)
            rot_tof = compute_tof_from_mask(rot_mask)

            rot_base_name = f"{base_name}_rot{r_idx}"
            cv2.imwrite(os.path.join(target_sub_dir, f"{rot_base_name}.jpg"), rot_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            np.save(os.path.join(target_sub_dir, f"{rot_base_name}_tof.npy"), rot_tof)
            total_saved += 1

            # Save photometric variants of this rotated pose (reusing rot_tof)
            for p_idx in range(1, PHOTOMETRIC_VARIANTS + 1):
                rot_p_img = generate_photometric_variant(rot_rgb, p_idx)
                r_p_name = f"{rot_base_name}_p{p_idx}.jpg"
                r_p_tof_name = f"{rot_base_name}_p{p_idx}_tof.npy"

                cv2.imwrite(os.path.join(target_sub_dir, r_p_name), rot_p_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                np.save(os.path.join(target_sub_dir, r_p_name), rot_tof)
                total_saved += 1

        if idx % 50 == 0 or idx == len(raw_images):
            print(f"Processed [{idx}/{len(raw_images)}] base samples -> {total_saved} total pairs generated.")

    print(f"\nStep 3 Complete: Full dataset assembled in '{DST_AUG_DIR}' ({total_saved} pairs total).")


if __name__ == "__main__":
    main()