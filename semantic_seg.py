import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import random
from unet import UNet

# Dataset personalizado
class FloodDataset(Dataset):
    def __init__(self, image_files, image_dir, mask_dir):
        self.image_files = image_files
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.resize = torchvision.transforms.Resize((128, 128))
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize([0.5]*3, [0.5]*3)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, image_file.replace(".jpg", ".png"))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.normalize(self.to_tensor(self.resize(image)))
        mask = self.resize(mask)
        mask = (self.to_tensor(mask) > 0).long().squeeze()

        return image, mask

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

# Preparación de datos
image_dir = "./data/Image"
mask_dir = "./data/Mask"
images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

random.seed(42)
random.shuffle(images)
split = int(0.8 * len(images))
train_files = images[:split]
val_files = images[split:]

train_dataset = FloodDataset(train_files, image_dir, mask_dir)
val_dataset = FloodDataset(val_files, image_dir, mask_dir)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
train_loss_list, val_loss_list = [], []
train_iou_list, val_iou_list = [], []

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

    train_loss_list.append(train_loss / len(train_loader))
    val_loss_list.append(val_loss / len(val_loader))
    train_iou_list.append(train_iou / len(train_loader))
    val_iou_list.append(val_iou / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_loss_list[-1]:.4f} | Val Loss: {val_loss_list[-1]:.4f} | "
          f"Train IoU: {train_iou_list[-1]:.4f} | Val IoU: {val_iou_list[-1]:.4f}")

# Gráficas
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss', marker='o')
plt.plot(val_loss_list, label='Validation Loss', marker='o')
plt.title('Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_iou_list, label='Train IoU', marker='o')
plt.plot(val_iou_list, label='Validation IoU', marker='o')
plt.title('Jaccard Index (IoU) por Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("resultados_mejorados.png")
plt.show()
