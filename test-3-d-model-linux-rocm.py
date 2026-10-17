import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_augmented_dataset" if os.path.isdir("gesturanet_augmented_dataset") else "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"    # Retained folder name; calibrated to 30 cm
WEIGHTS_PATH = "weights/gesturanet_4class_best.pth"

CALIBRATED_DISTANCE_M = 0.30   # 30.0 cm baseline

ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign"
]

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.20

# Modality Dropout & Auxiliary Supervision
RGB_DROPOUT_PROB = 0.30        # 30% of batches force pure ToF learning
AUX_LOSS_WEIGHT = 0.40         # Direct supervision on ToF depth branch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class GesturaNetPairedDataset(Dataset):
    """
    Loads paired (RGB image, 8x8 ToF .npy) files from the dataset directory.
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

        print(f"Loaded {len(self.samples)} paired (RGB, ToF) samples across {len(active_gestures)} classes from '{base_dir}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tof_path, label = self.samples[idx]

        # 1. RGB Image
        image = Image.open(img_path).convert("RGB")
        rgb_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)

        # 2. 8x8 ToF Depth Matrix (meters), zero-safe against NaNs
        depth_array = np.load(tof_path).astype(np.float32)
        depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0)  # Shape: (1, 8, 8)

        return rgb_tensor, depth_tensor, label


class GatedMultimodalGesturaNet(nn.Module):
    """
    Balanced Two-Stream Architecture with Gated Multimodal Fusion (GMU):
    - RGB Stream: MobileNetV3-Small -> 128-d projection
    - Depth Stream: Spatial Conv2D (preserves 8x8 geometry) -> 128-d projection
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
        # 8x8 -> Conv3x3 (32) -> Conv3x3 (64) -> MaxPool2x2 (4x4) -> 1024 -> 128
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: (B, 64, 4, 4)
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

        # Auxiliary Depth Classifier (forces depth backbone to learn independently)
        self.aux_depth_classifier = nn.Linear(128, num_classes)

    def forward(self, rgb, depth, return_aux=False):
        # RGB Stream -> (B, 128)
        f_rgb = self.rgb_pool(self.rgb_features(rgb)).flatten(1)
        emb_rgb = self.rgb_proj(f_rgb)

        # Depth Stream -> (B, 128)
        f_depth = self.depth_conv(depth)
        emb_depth = self.depth_proj(f_depth)

        # Gated Fusion (balance RGB and Depth)
        concat_feats = torch.cat((emb_rgb, emb_depth), dim=1)
        gate = self.gate_layer(concat_feats)
        fused_emb = gate * emb_rgb + (1.0 - gate) * emb_depth

        logits = self.classifier(fused_emb)

        if return_aux:
            aux_logits = self.aux_depth_classifier(emb_depth)
            return logits, aux_logits, gate

        return logits, gate


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
        print("Dataset contains 0 valid samples. Run Steps 1-B and 1-C first.")
        return None

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=3, 
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=3, 
        pin_memory=use_cuda
    )

    model = GatedMultimodalGesturaNet(num_classes=len(ACTIVE_GESTURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\nTraining Gated Two-Stream GesturaNet on Device: {DEVICE}")
    print(f"Modality Dropout Active: {int(RGB_DROPOUT_PROB * 100)}% of batches force pure ToF learning.")
    print(f"Auxiliary Depth Supervision Weight: {AUX_LOSS_WEIGHT}")
    print(f"Split: {train_size} train pairs | {val_size} validation pairs")
    print("----------------------------------------------------------------------------------------")

    best_val_acc = 0.0
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for rgb, depth, targets in train_loader:
            rgb, depth, targets = rgb.to(DEVICE), depth.to(DEVICE), targets.to(DEVICE)

            # Modality Dropout: randomly blackout RGB so the network must rely on ToF
            if random.random() < RGB_DROPOUT_PROB:
                rgb = torch.zeros_like(rgb)

            optimizer.zero_grad()
            outputs, aux_outputs, _ = model(rgb, depth, return_aux=True)

            loss_main = criterion(outputs, targets)
            loss_aux = criterion(aux_outputs, targets)
            total_loss = loss_main + (AUX_LOSS_WEIGHT * loss_aux)

            total_loss.backward()
            optimizer.step()

            train_loss += loss_main.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        depth_only_correct = 0

        with torch.no_grad():
            for rgb, depth, targets in val_loader:
                rgb, depth, targets = rgb.to(DEVICE), depth.to(DEVICE), targets.to(DEVICE)

                # Standard Fused Evaluation
                outputs, _ = model(rgb, depth)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()

                # Depth-Only Test: Zero out RGB to measure pure ToF performance
                black_rgb = torch.zeros_like(rgb)
                depth_only_outputs, _ = model(black_rgb, depth)
                depth_only_preds = depth_only_outputs.argmax(dim=1)
                depth_only_correct += (depth_only_preds == targets).sum().item()

                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        depth_only_acc = (depth_only_correct / val_total) * 100

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"| Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.1f}% "
              f"| Val Acc: {epoch_val_acc:.1f}% "
              f"| Depth-Only Acc (Blackout RGB): {depth_only_acc:.1f}%")

        # Save Best Model Weights
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)

    print(f"\nTraining Complete: Best model saved to {WEIGHTS_PATH} (Fused Val Acc: {best_val_acc:.1f}%)")
    return model


def predict_sample(model, image_path, tof_path):
    """
    Inference helper: Evaluates a sample pair and returns a sorted ranking.
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
        depth_array = np.nan_to_num(depth_array, nan=0.0)
    else:
        depth_array = np.zeros((8, 8), dtype=np.float32)
        depth_array[2:6, 2:6] = CALIBRATED_DISTANCE_M

    depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, gate = model(rgb_tensor, depth_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    results = [
        {"gesture": ACTIVE_GESTURES[i], "confidence": round(float(probs[i]), 4)}
        for i in range(len(ACTIVE_GESTURES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


if __name__ == "__main__":
    trained_model = train()

    first_test_img = glob.glob(f"{BASE_DATASET_DIR}/{ACTIVE_GESTURES[0]}/{TARGET_DISTANCE_DIR}/img_*.jpg")
    if first_test_img and trained_model is not None:
        img_p = first_test_img[0]
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        tof_p = os.path.join(os.path.dirname(img_p), f"{base_name}_tof.npy")

        print(f"\n--- Verification Inference on Sample: {os.path.basename(img_p)} ---")
        predictions = predict_sample(trained_model, img_p, tof_p)
        for rank, item in enumerate(predictions, 1):
            print(f" {rank}. {item['gesture']:<15} : {item['confidence'] * 100:.2f}%")