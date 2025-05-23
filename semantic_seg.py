import os
import torch
import torchvision
import PIL
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from unet import UNet
import random

# Separación manual 80/20
images = sorted(os.listdir("./data/Image/"))
random.seed(42)
random.shuffle(images)
split = int(0.8 * len(images))
train_images = images[:split]
val_images = images[split:]

def load_data(image_list):
    image_tensor = []
    mask_tensor = []
    for image in image_list:
        img = Image.open(f'./data/Image/{image}').convert("RGB")
        mask = Image.open(f'./data/Mask/{image.replace(".jpg", ".png")}').convert("L")

        img = torchvision.transforms.functional.resize(img, (128, 128))
        mask = torchvision.transforms.functional.resize(mask, (128, 128))

        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.Normalize([0.5]*3, [0.5]*3)(img)

        mask = torch.tensor(torch.as_tensor(mask) > 0, dtype=torch.long)

        image_tensor.append(img.unsqueeze(0))
        mask_tensor.append(mask.unsqueeze(0))

    return torch.cat(image_tensor), torch.cat(mask_tensor)

# Carga
x_train, y_train = load_data(train_images)
x_val, y_val = load_data(val_images)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train, y_train = x_train.to(device), y_train.to(device)
x_val, y_val = x_val.to(device), y_val.to(device)

train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(list(zip(x_val, y_val)), batch_size=16)

model = UNet(n_channels=3, n_classes=2).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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

train_loss_list, val_loss_list = [], []
train_iou_list, val_iou_list = [], []

for epoch in range(20):
    model.train()
    train_loss, train_iou = 0, 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optim.step()
        optim.zero_grad()
        train_loss += loss.item()
        train_iou += compute_iou(pred.detach(), mask)

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            val_loss += criterion(pred, mask).item()
            val_iou += compute_iou(pred, mask)

    train_loss_list.append(train_loss / len(train_loader))
    val_loss_list.append(val_loss / len(val_loader))
    train_iou_list.append(train_iou / len(train_loader))
    val_iou_list.append(val_iou / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_loss_list[-1]:.4f} | Val Loss: {val_loss_list[-1]:.4f} | "
          f"Train IoU: {train_iou_list[-1]:.4f} | Val IoU: {val_iou_list[-1]:.4f}")

# Graficar
epochs = list(range(1, 21))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, label='Train Loss', marker='o')
plt.plot(epochs, val_loss_list, label='Validation Loss', marker='o')
plt.title('Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_iou_list, label='Train IoU', marker='o')
plt.plot(epochs, val_iou_list, label='Validation IoU', marker='o')
plt.title('Jaccard Index (IoU) por Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("graficas_entrenamiento.png")
plt.show()
