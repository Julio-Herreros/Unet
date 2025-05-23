import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet

# Dataset personalizado
class FloodSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img = torchvision.transforms.functional.resize(img, (128, 128))
        mask = torchvision.transforms.functional.resize(mask, (128, 128))

        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)(img)

        mask = torch.tensor((torch.as_tensor(mask) > 0), dtype=torch.long)

        return img, mask

# IoU
def compute_iou(pred, target, num_classes=2):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(1.0)  # Asume 100% si no hay clase presente
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)

# Carga de datos
image_dir = './data/Image/'
mask_dir = './data/Mask/'
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_paths = sorted([os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in os.listdir(image_dir)])

train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

train_dataset = FloodSegmentationDataset(train_imgs, train_masks)
val_dataset = FloodSegmentationDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses, train_ious, val_ious = [], [], [], []

# Entrenamiento
for epoch in range(30):
    model.train()
    epoch_loss, epoch_iou = 0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_iou += compute_iou(outputs.detach(), masks)

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, masks).item()
            val_iou += compute_iou(outputs, masks)

    n_train = len(train_loader)
    n_val = len(val_loader)
    train_losses.append(epoch_loss / n_train)
    val_losses.append(val_loss / n_val)
    train_ious.append(epoch_iou / n_train)
    val_ious.append(val_iou / n_val)

    print(f"Epoch {epoch+1}: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}, "
          f"Train IoU {train_ious[-1]:.4f}, Val IoU {val_ious[-1]:.4f}")

# Gráficas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_ious, label='Train IoU', marker='o')
plt.plot(val_ious, label='Validation IoU', marker='o')
plt.title('Jaccard Index (IoU) por Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("resultados_entrenamiento.png")
plt.show()
