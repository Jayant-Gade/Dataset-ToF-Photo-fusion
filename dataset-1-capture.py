import cv2
import os
import re
import json
import time
import numpy as np
from datetime import datetime

# ================= GESTURE ARRAY CONFIGURATION =================
GESTURE_NAMES = [
    # Standard & Social Gestures
    "closed_fist",
    "open_palm",
    "thumbs_up",
    "thumbs_down",
    "victory",
    "ok_sign",
    "i_love_you",
    "rock_on",
    "call_me",
    "gun_shape",

    # Directional & Pointing
    "pointing_up",
    "pointing_down",
    "pointing_left",
    "pointing_right",
    "two_finger_point",

    # Number Counting (Single Hand)
    "count_1",
    "count_2",
    "count_3",
    "count_4",
    "count_5",

    # Precision & Manipulation
    "pinch_index",
    "pinch_middle",
    "pinch_all",
    "grab",
    "flat_hand_horizontal",
    "hand_edge",

    # Single-Hand Motions / Action States
    "swipe_left",
    "swipe_right",
    "swipe_up",
    "swipe_down",
    "pinch_in",
    "pinch_out",
    "wave",
    "beckon",
    "push",
    "pull",
    "circle_clockwise",
    "circle_counter_clockwise",
]

DISTANCE_CM = 3                # Measured distance in cm
BASE_DATASET_DIR = "gesturanet_raw_dataset"

CAMERA_INDEX = 0               # Default webcam
VIEWPORT_SIZE = 640            # Square viewport dimensions (px)
PANEL_WIDTH = 300              # Sidebar control width (px)
BURST_COUNT = 10               # Number of frames in burst mode
BURST_DELAY = 0.15             # Delay between burst shots (seconds)
# ==============================================================


class SequenceManager:
    """Monitors directory continuity, counts total images, and tracks missing slots."""
    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.pattern = re.compile(r"^img_(\d+)\.jpg$")
        self.existing_ids = set()
        self.missing_queue = []
        self.next_append_id = 1
        self.refresh()

    def refresh(self):
        self.existing_ids = set()
        if os.path.exists(self.target_dir):
            for fname in os.listdir(self.target_dir):
                match = self.pattern.match(fname)
                if match:
                    self.existing_ids.add(int(match.group(1)))

        if self.existing_ids:
            max_id = max(self.existing_ids)
            all_expected = set(range(1, max_id + 1))
            self.missing_queue = sorted(list(all_expected - self.existing_ids))
            self.next_append_id = max_id + 1
        else:
            self.missing_queue = []
            self.next_append_id = 1

    @property
    def total_count(self):
        return len(self.existing_ids)

    def peek_next(self):
        if self.missing_queue:
            return self.missing_queue[0], True
        return self.next_append_id, False

    def get_next_slot(self):
        if self.missing_queue:
            slot = self.missing_queue.pop(0)
            self.existing_ids.add(slot)
            return slot, True
        slot = self.next_append_id
        self.existing_ids.add(slot)
        self.next_append_id += 1
        return slot, False


def crop_center_square(frame):
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
    return cv2.resize(cropped, (VIEWPORT_SIZE, VIEWPORT_SIZE), interpolation=cv2.INTER_AREA)


def draw_8x8_grid(image):
    """Draws 8x8 sensor alignment grid overlay."""
    overlay = image.copy()
    step = VIEWPORT_SIZE // 8
    for i in range(1, 8):
        pos = i * step
        cv2.line(overlay, (pos, 0), (pos, VIEWPORT_SIZE), (0, 255, 0), 1)
        cv2.line(overlay, (0, pos), (VIEWPORT_SIZE, pos), (0, 255, 0), 1)
    cv2.rectangle(overlay, (0, 0), (VIEWPORT_SIZE - 1, VIEWPORT_SIZE - 1), (0, 255, 0), 2)
    return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)


def load_metadata_dict(metadata_path):
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
                return {item["file_name"]: item for item in data}
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def save_metadata_dict(metadata_path, meta_dict):
    sorted_records = sorted(meta_dict.values(), key=lambda x: x["file_name"])
    with open(metadata_path, "w") as f:
        json.dump(sorted_records, f, indent=4)


# Interactive UI button bounds: (x1, y1, x2, y2)
BUTTONS = {
    "prev_gesture": (VIEWPORT_SIZE + 15, 60, VIEWPORT_SIZE + 140, 95),
    "next_gesture": (VIEWPORT_SIZE + 160, 60, VIEWPORT_SIZE + 285, 95),
    "capture":      (VIEWPORT_SIZE + 20, 290, VIEWPORT_SIZE + PANEL_WIDTH - 20, 345),
    "burst":        (VIEWPORT_SIZE + 20, 360, VIEWPORT_SIZE + PANEL_WIDTH - 20, 415),
    "grid":         (VIEWPORT_SIZE + 20, 430, VIEWPORT_SIZE + PANEL_WIDTH - 20, 485),
    "quit":         (VIEWPORT_SIZE + 20, 570, VIEWPORT_SIZE + PANEL_WIDTH - 20, 620),
}

action_flags = {
    "prev": False,
    "next": False,
    "capture": False,
    "burst": False,
    "toggle_grid": False,
    "quit": False
}


def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for action, (x1, y1, x2, y2) in BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "prev_gesture":
                    action_flags["prev"] = True
                elif action == "next_gesture":
                    action_flags["next"] = True
                elif action == "capture":
                    action_flags["capture"] = True
                elif action == "burst":
                    action_flags["burst"] = True
                elif action == "grid":
                    action_flags["toggle_grid"] = True
                elif action == "quit":
                    action_flags["quit"] = True


def draw_control_panel(panel, gesture, g_idx, total_g, distance, seq_mgr, grid_active):
    panel[:] = (28, 30, 33)  # Dark slate background

    # Title Bar
    cv2.putText(panel, "GESTURANET DATASET", (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(panel, "GENERATION STUDIO", (40, 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    #GENERATION STUDIO
    cv2.line(panel, (20, 42), (PANEL_WIDTH - 20, 42), (65, 65, 65), 1)

    def draw_btn(bbox, label, bg_color, text_color=(255, 255, 255), font_scale=0.48):
        bx1, by1, bx2, by2 = bbox[0] - VIEWPORT_SIZE, bbox[1], bbox[2] - VIEWPORT_SIZE, bbox[3]
        cv2.rectangle(panel, (bx1, by1), (bx2, by2), bg_color, -1)
        cv2.rectangle(panel, (bx1, by1), (bx2, by2), (180, 180, 180), 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = bx1 + (bx2 - bx1 - tw) // 2
        ty = by1 + (by2 - by1 + th) // 2
        cv2.putText(panel, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    # Navigation buttons
    draw_btn(BUTTONS["prev_gesture"], "< PREV (Left)", (60, 60, 60))
    draw_btn(BUTTONS["next_gesture"], "NEXT > (Right)", (60, 60, 60))

    # Current Gesture Info
    cv2.putText(panel, f"[{g_idx + 1}/{total_g}]", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    
    # Scale long gesture text to fit nicely
    font_scale = 0.55 if len(gesture) <= 15 else 0.42
    cv2.putText(panel, gesture, (20, 142), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 200), 1)
    cv2.putText(panel, f"Distance: {distance} cm", (20, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Dataset Status Card
    cv2.rectangle(panel, (15, 180), (PANEL_WIDTH - 15, 270), (42, 45, 50), -1)
    cv2.rectangle(panel, (15, 180), (PANEL_WIDTH - 15, 270), (70, 75, 80), 1)

    total_imgs = seq_mgr.total_count
    next_id, is_gap = seq_mgr.peek_next()
    gaps_left = len(seq_mgr.missing_queue)

    cv2.putText(panel, f"Images in folder: {total_imgs}", (25, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    
    if is_gap:
        gap_txt = f"Status: FILLING GAP ({gaps_left} left)"
        cv2.putText(panel, gap_txt, (25, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(panel, f"Next: img_{next_id:05d}.jpg", (25, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    else:
        cv2.putText(panel, "Status: Continuous sequence", (25, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 220, 120), 1)
        cv2.putText(panel, f"Next: img_{next_id:05d}.jpg", (25, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Action Buttons
    draw_btn(BUTTONS["capture"], "CAPTURE (Space)", (35, 135, 45), font_scale=0.52)
    draw_btn(BUTTONS["burst"], f"BURST {BURST_COUNT}x (B)", (165, 85, 20), font_scale=0.52)
    grid_col = (0, 125, 195) if grid_active else (60, 60, 60)
    grid_txt = "GRID: ACTIVE (G)" if grid_active else "GRID: OFF (G)"
    draw_btn(BUTTONS["grid"], grid_txt, grid_col, font_scale=0.52)
    draw_btn(BUTTONS["quit"], "QUIT (Q)", (45, 45, 180), font_scale=0.52)


def main():
    gesture_idx = 0
    total_gestures = len(GESTURE_NAMES)

    # Directory & Sequence Management setup
    def get_current_dir(g_name):
        return os.path.join(BASE_DATASET_DIR, g_name, f"{int(DISTANCE_CM)}cm")

    current_gesture = GESTURE_NAMES[gesture_idx]
    target_dir = get_current_dir(current_gesture)
    os.makedirs(target_dir, exist_ok=True)

    seq_mgr = SequenceManager(target_dir)
    metadata_path = os.path.join(target_dir, "metadata.json")
    meta_dict = load_metadata_dict(metadata_path)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Unable to connect to camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    window_name = "GesturaNet Dataset generation Studio"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    show_grid = True
    full_frame = np.zeros((VIEWPORT_SIZE, VIEWPORT_SIZE + PANEL_WIDTH, 3), dtype=np.uint8)
    sidebar_panel = np.zeros((VIEWPORT_SIZE, PANEL_WIDTH, 3), dtype=np.uint8)

    def switch_gesture(new_index):
        nonlocal gesture_idx, current_gesture, target_dir, seq_mgr, metadata_path, meta_dict
        gesture_idx = new_index % total_gestures
        current_gesture = GESTURE_NAMES[gesture_idx]
        target_dir = get_current_dir(current_gesture)
        os.makedirs(target_dir, exist_ok=True)
        seq_mgr = SequenceManager(target_dir)
        metadata_path = os.path.join(target_dir, "metadata.json")
        meta_dict = load_metadata_dict(metadata_path)
        print(f"Switched to: [{gesture_idx + 1}/{total_gestures}] {current_gesture} | Existing count: {seq_mgr.total_count}")

    def save_single_frame(clean_frame):
        slot_id, is_gap = seq_mgr.get_next_slot()
        filename = f"img_{slot_id:05d}.jpg"
        save_path = os.path.join(target_dir, filename)

        cv2.imwrite(save_path, clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        meta_dict[filename] = {
            "file_name": filename,
            "gesture": current_gesture,
            "distance_cm": DISTANCE_CM,
            "timestamp": datetime.now().isoformat(),
            "crop_dimensions": [VIEWPORT_SIZE, VIEWPORT_SIZE],
            "raw_camera_resolution": [actual_w, actual_h],
            "camera_fps": fps,
            "filled_gap": is_gap
        }
        save_metadata_dict(metadata_path, meta_dict)
        tag = "[FILLED GAP]" if is_gap else "[NEW]"
        print(f"Saved {tag}: {current_gesture}/{filename}")

    print("\n--- GesturaNet Dataset Collector Active ---")
    print(f"Initial: {current_gesture} ({seq_mgr.total_count} images existing)")
    print("Use Left/Right arrow keys (or A/D keys) to cycle gestures.")
    print("-------------------------------------------\n")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to acquire frame.")
            break

        clean_square = crop_center_square(raw_frame)
        viewport_display = clean_square.copy()

        if show_grid:
            viewport_display = draw_8x8_grid(viewport_display)

        # Draw UI
        draw_control_panel(sidebar_panel, current_gesture, gesture_idx, total_gestures, DISTANCE_CM, seq_mgr, show_grid)
        full_frame[:, :VIEWPORT_SIZE] = viewport_display
        full_frame[:, VIEWPORT_SIZE:] = sidebar_panel

        cv2.imshow(window_name, full_frame)
        
        # cv2.waitKeyEx captures multi-byte directional arrow keys across OS platforms
        key = cv2.waitKeyEx(1)

        # Handle Quit
        if action_flags["quit"] or key in (ord('q'), ord('Q'), 27):
            break

        # Handle Previous Gesture: Left Arrow (Linux: 65361, Windows: 2424832, Common: 81), or 'A', or '< PREV' button
        if action_flags["prev"] or key in (65361, 2424832, 81, ord('a'), ord('A'), ord(',')):
            switch_gesture(gesture_idx - 1)
            action_flags["prev"] = False

        # Handle Next Gesture: Right Arrow (Linux: 65363, Windows: 2555904, Common: 83), or 'D', or 'NEXT >' button
        if action_flags["next"] or key in (65363, 2555904, 83, ord('d'), ord('D'), ord('.')):
            switch_gesture(gesture_idx + 1)
            action_flags["next"] = False

        # Handle Grid Toggle
        if action_flags["toggle_grid"] or key in (ord('g'), ord('G')):
            show_grid = not show_grid
            action_flags["toggle_grid"] = False

        # Handle Single Capture
        if action_flags["capture"] or key == 32:  # SPACE
            save_single_frame(clean_square)
            action_flags["capture"] = False

        # Handle Burst Capture
        if action_flags["burst"] or key in (ord('b'), ord('B')):
            print(f"Burst capturing {BURST_COUNT} images for {current_gesture}...")
            for _ in range(BURST_COUNT):
                ret_b, raw_b = cap.read()
                if ret_b:
                    clean_b = crop_center_square(raw_b)
                    save_single_frame(clean_b)
                time.sleep(BURST_DELAY)
            print("Burst capture complete.")
            action_flags["burst"] = False

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession completed successfully.")


if __name__ == "__main__":
    main()