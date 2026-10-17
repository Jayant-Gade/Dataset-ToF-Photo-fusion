import os
import glob
import random
import shutil
import cv2
import numpy as np

# ================= CONFIGURATION =================
DATASET_DIR = "gesturanet_augmented_dataset"
TARGET_DISTANCE_DIR = "3cm"

# Photometric variation schedule (intensity factors from 5% to 90%)
# Negative factors = Darkening / Underexposure
# Positive factors = Brightening / Glare
AUGMENTATION_LEVELS = [
    {"name": "subtle_dark",  "intensity": -0.05, "blur": False},
    {"name": "subtle_bright", "intensity":  0.05, "blur": False},
    {"name": "mod_dark",     "intensity": -0.25, "blur": False},
    {"name": "mod_bright",   "intensity":  0.25, "blur": False},
    {"name": "heavy_dark",   "intensity": -0.60, "blur": True},
    {"name": "heavy_bright",  "intensity":  0.60, "blur": False},
    {"name": "extreme_dark", "intensity": -0.90, "blur": True},   # Hand barely visible in shadows
    {"name": "extreme_bright", "intensity": 0.85, "blur": False},  # Near-washed out overexposure
]
# =================================================


def apply_photometrics(image_bgr, intensity, apply_blur):
    """
    Applies photometric changes across the spectrum (-0.90 to +0.85).
    Negative intensity dims the image down to near-black.
    Positive intensity washes the image out toward white.
    """
    img = image_bgr.astype(np.float32)

    if intensity < 0:
        # Darkening: multiplier scales from 0.95 down to 0.10
        alpha = 1.0 + intensity  # e.g., -0.90 -> alpha = 0.10
        beta = intensity * 40.0   # e.g., -0.90 -> beta = -36
        img = img * alpha + beta
    else:
        # Brightening: multiplier scales from 1.05 up to 1.80
        alpha = 1.0 + (intensity * 0.9)
        beta = intensity * 50.0
        img = img * alpha + beta

    # Apply subtle defocus blur on selected dark conditions (simulates low-light sensor drag)
    if apply_blur and random.random() > 0.3:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Subtle random ISO sensor noise
    noise_sigma = abs(intensity) * 6.0
    if noise_sigma > 1.0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    target_pattern = os.path.join(DATASET_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    all_images = sorted(glob.glob(target_pattern))

    # Select only base poses (orig and rot), ignoring existing photometric variants
    base_images = [
        p for p in all_images
        if ("_orig.jpg" in p or "_rot" in p) and not any(lvl["name"] in p for lvl in AUGMENTATION_LEVELS)
    ]

    if not base_images:
        print(f"No base images found in '{DATASET_DIR}'. Run Steps 1-B and 1-C first.")
        return

    print("--- Step 1-D: Wide-Spectrum Photometric Augmentation ---")
    print(f"Found {len(base_images)} base images with precomputed ToF files.")
    print(f"Generating {len(AUGMENTATION_LEVELS)} lighting variants (5% to 90% range) per frame...\n")

    total_created = 0

    for idx, img_path in enumerate(base_images, 1):
        folder = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        tof_path = os.path.join(folder, f"{base_name}_tof.npy")

        # Must have an existing valid ToF file
        if not os.path.isfile(tof_path):
            continue

        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            continue

        for lvl in AUGMENTATION_LEVELS:
            variant_name = lvl["name"]
            out_img_name = f"{base_name}_{variant_name}.jpg"
            out_tof_name = f"{base_name}_{variant_name}_tof.npy"

            out_img_path = os.path.join(folder, out_img_name)
            out_tof_path = os.path.join(folder, out_tof_name)

            # 1. Transform RGB pixels
            aug_bgr = apply_photometrics(raw_bgr, lvl["intensity"], lvl["blur"])
            cv2.imwrite(out_img_path, aug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            # 2. Clone identical ToF file without recomputing depth
            shutil.copyfile(tof_path, out_tof_path)
            total_created += 1

        if idx % 50 == 0 or idx == len(base_images):
            print(f"Processed [{idx}/{len(base_images)}] base poses -> {total_created} lighting pairs added.")

    print(f"\nStep 1-D Complete: Generated {total_created} photometric pairs.")
    print(f"Full dataset ready for model training in '{DATASET_DIR}'.")


if __name__ == "__main__":
    main()