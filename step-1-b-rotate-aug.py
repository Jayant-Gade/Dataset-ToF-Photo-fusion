import os
import glob
import random
import cv2

# ================= CONFIGURATION =================
SRC_RAW_DIR = "gesturanet_raw_dataset"
DST_AUG_DIR = "gesturanet_augmented_dataset"
TARGET_DISTANCE_DIR = "3cm"

# Number of rotated poses to create per raw image
NUM_ROTATIONS = 2

# Geometric limits (safe for all gesture classes)
MAX_ROTATION_DEG = 8.0     # Strictly +-8 degrees
SCALE_MIN = 0.96
SCALE_MAX = 1.04
TRANSLATION_RATIO = 0.04   # +-4% horizontal and vertical translation
# =================================================


def apply_safe_rotation(image_bgr):
    """Applies bounded rotation, scale, and translation without flipping axes."""
    angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    scale = random.uniform(SCALE_MIN, SCALE_MAX)

    h, w = image_bgr.shape[:2]
    max_dx = int(w * TRANSLATION_RATIO)
    max_dy = int(h * TRANSLATION_RATIO)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    return cv2.warpAffine(
        image_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def main():
    target_pattern = os.path.join(SRC_RAW_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    raw_images = sorted(glob.glob(target_pattern))

    if not raw_images:
        print(f"No source images found in '{SRC_RAW_DIR}'.")
        return

    print(f"--- Step 1-B: Geometric Rotation Augmenter ---")
    print(f"Source: '{SRC_RAW_DIR}' -> Destination: '{DST_AUG_DIR}'")
    print(f"Processing {len(raw_images)} captures ({NUM_ROTATIONS} rotated poses per image)...\n")

    total_created = 0

    for idx, img_path in enumerate(raw_images, 1):
        rel_path = os.path.relpath(img_path, SRC_RAW_DIR)
        sub_folder = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        target_dir = os.path.join(DST_AUG_DIR, sub_folder)
        os.makedirs(target_dir, exist_ok=True)

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue

        # 1. Save pristine original image into target dataset
        orig_out = os.path.join(target_dir, f"{base_name}_orig.jpg")
        cv2.imwrite(orig_out, raw_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        total_created += 1

        # 2. Save rotated variations
        for r_idx in range(1, NUM_ROTATIONS + 1):
            rot_img = apply_safe_rotation(raw_img)
            rot_out = os.path.join(target_dir, f"{base_name}_rot{r_idx}.jpg")
            cv2.imwrite(rot_out, rot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            total_created += 1

        if idx % 50 == 0 or idx == len(raw_images):
            print(f"Processed [{idx}/{len(raw_images)}] images -> {total_created} frames created.")

    print(f"\nStep 1-B Complete: Generated {total_created} images in '{DST_AUG_DIR}'.")


if __name__ == "__main__":
    main()