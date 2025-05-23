import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet  # usa aquí tu clase U-Net real

# Dataset
class FloodDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.resize = torchvision.transforms.Resize((128, 128))
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img = self.normalize(self.to_tensor(self.resize(img)))
        mask = torch.tensor(self.resize(mask) > 0, dtype=torch.long)
        return img, mask

# IoU
def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1)
    ious = []
    for cls in range(2):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum((1, 2))
        union = (pred_inds | target_inds).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou.mean().item())
    return sum(ious) / len(ious)

# Cargar rutas
image_dir = './data/Image/'
mask_dir = './data/Mask/'
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
image_paths = [os.path.join(image_dir, f) for f in images]
mask_paths = [os.path.join(mask_dir, f.replace('.jpg', '.png')) for f in images]

# División train/val
train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
train_dataset = FloodDataset(train_imgs, train_masks)
val_dataset = FloodDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses, train_ious, val_ious = [], [], [], []

# Entrenamiento
for epoch in range(20):
    model.train()
    train_loss, train_iou = 0, 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_iou += compute_iou(outputs.detach(), masks)

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, masks).item()
            val_iou += compute_iou(outputs, masks)

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    train_ious.append(train_iou / len(train_loader))
    val_ious.append(val_iou / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
          f"Train IoU: {train_ious[-1]:.4f} | Val IoU: {val_ious[-1]:.4f}")

# Gráficas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title('Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_ious, label='Train IoU', marker='o')
plt.plot(val_ious, label='Validation IoU', marker='o')
plt.title('Jaccard Index (IoU) por Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("mejores_resultados_entrenamiento.png")
plt.show()
