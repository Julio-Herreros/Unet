import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from unet import UNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# === 1. Carga de imágenes y máscaras ===
image_files = sorted(os.listdir("./data/Image"))
mask_files = sorted(os.listdir("./data/Mask"))

images = []
masks = []

for image_name, mask_name in zip(image_files, mask_files):
    # Imagen
    img = Image.open(f'./data/Image/{image}').convert("RGB")
    img = TF.resize(TF.pil_to_tensor(img), (100, 100)).float() / 255.
    images.append(img)

    # Máscara binaria
    mask = Image.open(f'./data/Mask/{mask}').convert("L")
    mask = TF.resize(TF.pil_to_tensor(mask), (100, 100))
    mask = (mask > 0).long().squeeze(0)
    masks.append(mask)

# Tensores
images_tensor = torch.stack(images)
masks_tensor = torch.stack(masks)

# === 2. División en entrenamiento y validación ===
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    images_tensor, masks_tensor, test_size=0.2, random_state=42
)

# Datasets y loaders
train_ds = TensorDataset(train_imgs, train_masks)
val_ds = TensorDataset(val_imgs, val_masks)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# === 3. Modelo ===
model = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# === 4. Entrenamiento ===
epochs = 20
train_loss_list, val_loss_list = [], []
train_iou_list, val_iou_list = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(out, dim=1)
        intersection = ((preds == 1) & (y == 1)).float().sum((1, 2))
        union = ((preds == 1) | (y == 1)).float().sum((1, 2))
        iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
        running_iou += iou

    train_loss_list.append(running_loss / len(train_loader))
    train_iou_list.append(running_iou / len(train_loader))

    # === Validación ===
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            intersection = ((preds == 1) & (y == 1)).float().sum((1, 2))
            union = ((preds == 1) | (y == 1)).float().sum((1, 2))
            iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
            val_iou += iou

    val_loss_list.append(val_loss / len(val_loader))
    val_iou_list.append(val_iou / len(val_loader))

    print(f"Epoch {epoch+1} | Train Loss: {train_loss_list[-1]:.4f} | Val Loss: {val_loss_list[-1]:.4f} | Train IoU: {train_iou_list[-1]:.4f} | Val IoU: {val_iou_list[-1]:.4f}")

# === 5. Gráficas ===
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
plt.savefig("metrics_segmentacion.png", dpi=300)
plt.show()
