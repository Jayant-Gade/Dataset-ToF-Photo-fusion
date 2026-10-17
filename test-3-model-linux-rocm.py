import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

# ================= CONFIGURATION =================
BASE_DATASET_DIR = "gesturanet_raw_dataset"
TARGET_DISTANCE_DIR = "3cm"
DISTANCE_METERS = 0.03         # 3 cm in meters
NOISE_PERCENT = 0.05           # 5% sensor noise

# Subset of gestures to train for testing
ACTIVE_GESTURES = [
    "open_palm",
    "closed_fist",
    "thumbs_up",
    "ok_sign",
    "pinch_index"
]

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================


class GesturaNetDataset(Dataset):
    """Loads RGB crops and synthesizes constant 8x8 distance arrays with noise."""
    def __init__(self, base_dir, active_gestures, dist_folder, dist_meters, noise_factor, transform=None):
        self.samples = []
        self.transform = transform
        self.dist_meters = dist_meters
        self.noise_factor = noise_factor
        self.label_to_idx = {name: idx for idx, name in enumerate(active_gestures)}

        for gesture_name in active_gestures:
            folder_path = os.path.join(base_dir, gesture_name, dist_folder)
            if not os.path.isdir(folder_path):
                print(f"Warning: Directory not found: {folder_path}")
                continue

            img_paths = glob.glob(os.path.join(folder_path, "img_*.jpg"))
            for path in img_paths:
                self.samples.append((path, self.label_to_idx[gesture_name]))

        print(f"Found {len(self.samples)} images across {len(active_gestures)} active classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            rgb_tensor = self.transform(image)
        else:
            rgb_tensor = transforms.ToTensor()(image)

        # Generate 8x8 depth array with 5% Gaussian noise (B, 1, 8, 8)
        base_depth = torch.full((1, 8, 8), self.dist_meters, dtype=torch.float32)
        noise = torch.randn(1, 8, 8) * (self.dist_meters * self.noise_factor)
        depth_tensor = torch.clamp(base_depth + noise, min=0.02, max=4.0)

        return rgb_tensor, depth_tensor, label


class TwoStreamGesturaNet(nn.Module):
    """Lightweight RGB + 8x8 ToF Mid-Level Fusion Network."""
    def __init__(self, num_classes=5):
        super().__init__()
        # Stream 1: RGB Backbone (MobileNetV3-Small)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_rgb = mobilenet_v3_small(weights=weights)
        self.rgb_features = base_rgb.features
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Stream 2: 8x8 ToF Depth Backbone
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Fusion Head: MobileNetV3 pool out = 576, Depth pool out = 32
        self.classifier = nn.Sequential(
            nn.Linear(576 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
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

    dataset = GesturaNetDataset(
        BASE_DATASET_DIR,
        ACTIVE_GESTURES,
        TARGET_DISTANCE_DIR,
        DISTANCE_METERS,
        NOISE_PERCENT,
        transform=rgb_transforms
    )

    if len(dataset) == 0:
        print("Dataset empty. Capture frames first using the collector script.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = TwoStreamGesturaNet(num_classes=len(ACTIVE_GESTURES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"Starting training on device: {DEVICE}")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb, depth, targets in loader:
            rgb, depth, targets = rgb.to(DEVICE), depth.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(rgb, depth)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    os.makedirs("weights", exist_ok=True)
    save_path = "weights/gesturanet_prototype.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel weights saved to: {save_path}")

    return model


def predict_single_frame(model, image_path, distance_meters=0.03):
    """Inference helper: Takes an image and a distance scalar, outputs sorted matching scores."""
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    rgb_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Constant sensor array emulation
    depth_tensor = torch.full((1, 1, 8, 8), distance_meters, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(rgb_tensor, depth_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Create ranked dictionary structure
    results = [
        {"gesture": ACTIVE_GESTURES[i], "score": round(float(probabilities[i]), 4)}
        for i in range(len(ACTIVE_GESTURES))
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    trained_model = train()

    # Quick test on first available sample
    first_sample = glob.glob(f"{BASE_DATASET_DIR}/{ACTIVE_GESTURES[0]}/{TARGET_DISTANCE_DIR}/img_*.jpg")
    if first_sample and trained_model is not None:
        print("\n--- Test Inference on Sample Frame ---")
        scores = predict_single_frame(trained_model, first_sample[0], DISTANCE_METERS)
        for rank, entry in enumerate(scores, 1):
            print(f" {rank}. {entry['gesture']:<16} : {entry['score']:.4f}")