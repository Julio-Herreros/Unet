import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from unet import UNet
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar imágenes y máscaras
image_tensor = []
mask_tensor = []

images = os.listdir('./data/Image/')
for image in images:
    img = Image.open(f'./data/Image/{image}').convert('RGB')
    img_tensor = torchvision.transforms.functional.pil_to_tensor(img)
    img_tensor = torchvision.transforms.functional.resize(img_tensor, (100, 100))
    img_tensor = img_tensor.clone().detach().float() / 255.
    image_tensor.append(img_tensor.unsqueeze(0))

    mask_name = image.replace('.jpg', '.png')
    mask = Image.open(f'./data/Mask/{mask_name}')
    mask_tensor_raw = torchvision.transforms.functional.pil_to_tensor(mask)
    mask_tensor_raw = torchvision.transforms.functional.resize(mask_tensor_raw, (100, 100))
    mask_bin = (mask_tensor_raw[:1] > 0).long()
    mask_one_hot = torch.nn.functional.one_hot(mask_bin[0], num_classes=2).permute(2, 0, 1)
    mask_tensor.append(mask_one_hot.unsqueeze(0).float())

image_tensor = torch.cat(image_tensor)
mask_tensor = torch.cat(mask_tensor)

print("Imágenes:", image_tensor.shape)
print("Máscaras:", mask_tensor.shape)

# División entrenamiento / validación
train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_tensor, mask_tensor, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.TensorDataset(train_imgs, train_masks)
val_dataset = torch.utils.data.TensorDataset(val_imgs, val_masks)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# Modelo y entrenamiento
unet = UNet(n_channels=3, n_classes=2).to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

train_loss_list = []
val_loss_list = []
train_iou_list = []
val_iou_list = []

for epoch in range(10):
    unet.train()
    running_loss = 0.

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = unet(imgs)
        loss = loss_fn(preds, torch.argmax(masks, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # IoU entrenamiento
    unet.eval()
    train_iou = []
    with torch.no_grad():
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = unet(imgs)
            pred_labels = torch.argmax(preds, dim=1)
            target_labels = torch.argmax(masks, dim=1)

            intersection = ((pred_labels == 1) & (target_labels == 1)).sum(dim=(1, 2))
            union = ((pred_labels == 1) | (target_labels == 1)).sum(dim=(1, 2))
            iou = (intersection.float() / (union.float() + 1e-6)).mean().item()
            train_iou.append(iou)

    # IoU validación
    val_loss = 0.
    val_iou = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = unet(imgs)
            loss = loss_fn(preds, torch.argmax(masks, dim=1))
            val_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1)
            target_labels = torch.argmax(masks, dim=1)

            intersection = ((pred_labels == 1) & (target_labels == 1)).sum(dim=(1, 2))
            union = ((pred_labels == 1) | (target_labels == 1)).sum(dim=(1, 2))
            iou = (intersection.float() / (union.float() + 1e-6)).mean().item()
            val_iou.append(iou)

    avg_train_iou = sum(train_iou) / len(train_iou)
    avg_val_iou = sum(val_iou) / len(val_iou)

    train_loss_list.append(running_loss)
    val_loss_list.append(val_loss)
    train_iou_list.append(avg_train_iou)
    val_iou_list.append(avg_val_iou)

    print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

# Gráficas
epochs = list(range(1, 11))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_list, label='Train Loss', marker='o')
plt.plot(epochs, val_loss_list, label='Validation Loss', marker='o')
plt.title('Loss por Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig("perdida-por-epoca.png")

plt.subplot(1, 2, 2)
plt.plot(epochs, train_iou_list, label='Train IoU', marker='o')
plt.plot(epochs, val_iou_list, label='Validation IoU', marker='o')
plt.title('Jaccard Index (IoU) por Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.grid(True)
plt.legend()
plt.title('Precisión-por-época.png')

plt.tight_layout()
plt.savefig("resultados_entrenamiento.png")
plt.show()
