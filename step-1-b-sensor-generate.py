import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"

# Physical Parameters (in meters)
HAND_DISTANCE_M = 0.03         # 3 cm physical ground truth
WALL_DISTANCE_M = 1.20         # Measured distance to background white wall
NOISE_PERCENT = 0.015          # 1.5% Gaussian sensor noise (within 1-2% target)
MIN_SENSOR_RANGE_M = 0.02      # VL53L8CX minimum physical range (20 mm)
MAX_SENSOR_RANGE_M = 4.00      # VL53L8CX maximum physical range (4000 mm)

# SPAD Activation Threshold
# Return photon intensity scales with 1/d^2. Because the hand at 3 cm is ~1600x
# more reflective than the wall at 1.2 m, even partial coverage triggers the hand return.
ZONE_ACTIVATION_THRESHOLD = 0.15   # 15% pixel occupancy triggers hand distance

# Visual Quality Control
GENERATE_PREVIEWS = True       # Saves side-by-side RGB vs 8x8 heatmap previews
# =================================================


def segment_hand_white_background(image_bgr):
    """
    Extracts a high-precision binary mask of the hand against a white wall.
    Combines Luminance/Saturation filtering with Otsu thresholding and morphological closing.
    """
    # Convert to HSV and LAB color spaces
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

    # White walls have high luminance and near-zero saturation
    saturation = hsv[:, :, 1]
    l_channel = lab[:, :, 0]

    # Mask non-white regions (saturation > 25 or brightness drop from white)
    _, sat_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)
    _, l_mask = cv2.threshold(l_channel, 215, 255, cv2.THRESH_BINARY_INV)

    combined_mask = cv2.bitwise_or(sat_mask, l_mask)

    # Morphological cleanup to eliminate specular sensor noise and fill hand interior
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    # Retain only the largest connected component (the hand)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(cleaned)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > 1000:  # Ignore micro noise
            cv2.drawContours(hand_mask, [largest_cnt], -1, 255, -1)

    return hand_mask


def compute_8x8_tof_depth(hand_mask, hand_d_m, wall_d_m, noise_ratio):
    """
    Projects the 8x8 VL53L8CX SPAD array onto the binary mask and computes
    stochastically noised physical depth values per zone.
    """
    h, w = hand_mask.shape
    cell_h = h // 8
    cell_w = w // 8

    depth_matrix = np.full((8, 8), wall_d_m, dtype=np.float32)
    coverage_matrix = np.zeros((8, 8), dtype=np.float32)

    for row in range(8):
        for col in range(8):
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w

            zone_patch = hand_mask[y1:y2, x1:x2]
            coverage = np.count_nonzero(zone_patch) / (cell_h * cell_w)
            coverage_matrix[row, col] = coverage

            if coverage >= ZONE_ACTIVATION_THRESHOLD:
                # Target lock on hand
                base_val = hand_d_m
            elif coverage > 0.03:
                # Edge transition: linear blending between hand and background
                t = (coverage - 0.03) / (ZONE_ACTIVATION_THRESHOLD - 0.03)
                base_val = (1.0 - t) * wall_d_m + t * hand_d_m
            else:
                # Pure background wall
                base_val = wall_d_m

            # Add calibrated Gaussian sensor noise (1% to 2%)
            noise = np.random.normal(loc=0.0, scale=base_val * noise_ratio)
            noisy_val = np.clip(base_val + noise, MIN_SENSOR_RANGE_M, MAX_SENSOR_RANGE_M)
            depth_matrix[row, col] = noisy_val

    return depth_matrix, coverage_matrix


def save_visual_preview(rgb_img, depth_matrix, save_path):
    """Generates an inspection plot comparing RGB crop, 8x8 overlay, and the depth heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # RGB with 8x8 grid overlay
    display_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    h, w, _ = display_rgb.shape
    step_y, step_x = h / 8, w / 8
    for i in range(1, 8):
        axes[0].axvline(i * step_x, color='lime', linewidth=0.8, alpha=0.7)
        axes[0].axhline(i * step_y, color='lime', linewidth=0.8, alpha=0.7)
    axes[0].imshow(display_rgb)
    axes[0].set_title("Input RGB (8x8 FoV)", fontsize=10)
    axes[0].axis('off')

    # Depth Heatmap (meters)
    im = axes[1].imshow(depth_matrix, cmap='viridis_r', vmin=0.02, vmax=WALL_DISTANCE_M)
    for r in range(8):
        for c in range(8):
            val = depth_matrix[r, c]
            txt = f"{val*100:.0f}"  # display in cm
            axes[1].text(c, r, txt, ha='center', va='center',
                         color='white' if val < (WALL_DISTANCE_M / 2) else 'black',
                         fontsize=7, fontweight='bold')
    axes[1].set_title("Synthetic ToF Depth (cm)", fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Meters")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    target_pattern = os.path.join(BASE_DATASET_DIR, "*", TARGET_DISTANCE_DIR, "img_*.jpg")
    image_paths = sorted(glob.glob(target_pattern))

    if not image_paths:
        print(f"No captured images found matching: {target_pattern}")
        return

    print(f"Found {len(image_paths)} images to process for {TARGET_DISTANCE_DIR}...")
    success_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        folder = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        depth_out_path = os.path.join(folder, f"{base_name}_tof.npy")

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # 1. High-accuracy foreground segmentation
        mask = segment_hand_white_background(frame)

        # 2. Physics-based 8x8 sensor simulation
        depth_8x8, _ = compute_8x8_tof_depth(
            mask,
            HAND_DISTANCE_M,
            WALL_DISTANCE_M,
            NOISE_PERCENT
        )

        # 3. Save raw matrix (8x8 float32 array in meters)
        np.save(depth_out_path, depth_8x8)

        # 4. Optional visual audit preview
        if GENERATE_PREVIEWS:
            preview_dir = os.path.join(folder, "previews")
            os.makedirs(preview_dir, exist_ok=True)
            preview_path = os.path.join(preview_dir, f"{base_name}_preview.jpg")
            save_visual_preview(frame, depth_8x8, preview_path)

        success_count += 1
        if idx % 25 == 0 or idx == len(image_paths):
            print(f"Processed [{idx}/{len(image_paths)}] frames -> {depth_out_path}")

    print(f"\nStep 2 complete: Successfully generated {success_count} ToF depth matrices (.npy).")


if __name__ == "__main__":
    main()