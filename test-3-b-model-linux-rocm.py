import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"

# 4 Active target gestures
ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign"
]

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.20  # 20% validation split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class GesturaNetPairedDataset(Dataset):
    """
    Loads paired (RGB image, 8x8 ToF .npy) files generated from Steps 1 and 2.
    """
    def __init__(self, base_dir, active_gestures, dist_folder, transform=None):
        self.samples = []
        self.transform = transform
        self.label_to_idx = {name: idx for idx, name in enumerate(active_gestures)}

        for gesture_name in active_gestures:
            folder_path = os.path.join(base_dir, gesture_name, dist_folder)
            if not os.path.isdir(folder_path):
                print(f"Directory not found: {folder_path}")
                continue

            img_paths = sorted(glob.glob(os.path.join(folder_path, "img_*.jpg")))
            for img_p in img_paths:
                base_name = os.path.splitext(os.path.basename(img_p))[0]
                tof_p = os.path.join(folder_path, f"{base_name}_tof.npy")

                if os.path.isfile(tof_p):
                    self.samples.append((img_p, tof_p, self.label_to_idx[gesture_name]))
                else:
                    print(f"Skipping {img_p} (Missing: {base_name}_tof.npy)")

        print(f"Loaded {len(self.samples)} paired (RGB, ToF) samples across {len(active_gestures)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tof_path, label = self.samples[idx]

        # 1. Load RGB Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            rgb_tensor = self.transform(image)
        else:
            rgb_tensor = transforms.ToTensor()(image)

        # 2. Load 8x8 ToF Depth Matrix (in meters)
        depth_array = np.load(tof_path).astype(np.float32)  # shape: (8, 8)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # shape: (1, 8, 8)

        return rgb_tensor, depth_tensor, label


class TwoStreamGesturaNet(nn.Module):
    """
    Multimodal Mid-Level Fusion Network:
    - RGB: MobileNetV3-Small backbone (160x160x3) -> 576-d vector[cite: 1]
    - Depth: 8x8x1 Mini-CNN -> 32-d vector[cite: 1]
    - Classifier: Concatenation (608-d) -> 64 -> 4 classes[cite: 1]
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB Stream
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_rgb = mobilenet_v3_small(weights=weights)
        self.rgb_features = base_rgb.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 8x8 ToF Depth Stream
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Multimodal Fusion Head (576 channels from RGB + 32 from Depth)
        self.classifier = nn.Sequential(
            nn.Linear(576 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, rgb, depth):
        # rgb: (B, 3, 160, 160), depth: (B, 1, 8, 8)
        x_rgb = self.rgb_pool(self.rgb_features(rgb)).flatten(1)
        x_depth = self.depth_branch(depth)
        
        fused = torch.cat((x_rgb, x_depth), dim=1)
        return self.classifier(fused)


def train():
    rgb_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = GesturaNetPairedDataset(
        BASE_DATASET_DIR,
        ACTIVE_GESTURES,
        TARGET_DISTANCE_DIR,
        transform=rgb_transforms
    )

    if len(full_dataset) == 0:
        print("Dataset contains 0 valid samples. Run Step 1 and Step 2 first.")
        return None

    # Train / Validation Split
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = TwoStreamGesturaNet(num_classes=len(ACTIVE_GESTURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\nTraining on: {DEVICE}")
    print(f"Split: {train_size} train frames | {val_size} validation frames")
    print("---------------------------------------------------------")

    best_val_acc = 0.0
    os.makedirs("weights", exist_ok=True)
    best_weights_path = "weights/gesturanet_4class_best.pth"

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for rgb, depth, targets in train_loader:
            rgb, depth, targets = rgb.to(DEVICE), depth.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for rgb, depth, targets in val_loader:
                rgb, depth, targets = rgb.to(DEVICE), depth.to(DEVICE), targets.to(DEVICE)
                outputs = model(rgb, depth)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = (val_loss / val_total) if val_total > 0 else 0.0
        epoch_val_acc = ((val_correct / val_total) * 100) if val_total > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.1f}% "
              f"| Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.1f}%")

        # Save Best Model Only
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"\nBest model saved to: {best_weights_path} with {best_val_acc:.1f}% validation accuracy.")
    return model


def predict_sample(model, image_path, tof_path):
    """
    Evaluates a sample pair and returns a sorted list of gesture confidence scores.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    rgb_tensor = transform(img).unsqueeze(0).to(DEVICE)

    if os.path.exists(tof_path):
        depth_array = np.load(tof_path).astype(np.float32)
    else:
        # Fallback to simulated 3cm array if .npy not specified
        depth_array = np.full((8, 8), 0.03, dtype=np.float32)

    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(rgb_tensor, depth_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    results = [
        {"gesture": ACTIVE_GESTURES[i], "confidence": round(float(probs[i]), 4)}
        for i in range(len(ACTIVE_GESTURES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


if __name__ == "__main__":
    trained_model = train()

    # Verification run on the first available sample
    first_test_img = glob.glob(f"{BASE_DATASET_DIR}/{ACTIVE_GESTURES[0]}/{TARGET_DISTANCE_DIR}/img_*.jpg")
    if first_test_img and trained_model is not None:
        img_p = first_test_img[0]
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        tof_p = os.path.join(os.path.dirname(img_p), f"{base_name}_tof.npy")

        print(f"\n--- Model Prediction on Sample: {img_p} ---")
        predictions = predict_sample(trained_model, img_p, tof_p)
        for rank, item in enumerate(predictions, 1):
            print(f" {rank}. {item['gesture']:<15} : {item['confidence']:.4f}")