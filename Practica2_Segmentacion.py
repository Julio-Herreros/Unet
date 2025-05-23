import torchvision
import torch
import PIL
from PIL import Image
from unet import UNet
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    images = sorted(os.listdir("./data/Image/"))
    masks = sorted(os.listdir("./data/Mask/"))

    image_tensor = []
    mask_tensor = []

    for image in images:
        img = PIL.Image.open(f'./data/Image/{image}').convert("RGB")
        img_tensor = torchvision.transforms.functional.pil_to_tensor(img)
        img_tensor = torchvision.transforms.functional.resize(img_tensor, (100, 100))
        img_tensor = img_tensor.float() / 255.
        image_tensor.append(img_tensor)

        mask_name = image.replace('.jpg', '.png')
        mask = PIL.Image.open(f'./data/Mask/{mask_name}').convert("L")
        mask_tensor_raw = torchvision.transforms.functional.pil_to_tensor(mask)
        mask_tensor_resized = torchvision.transforms.functional.resize(mask_tensor_raw, (100, 100))
        mask_tensor_bin = (mask_tensor_resized > 0).long().squeeze(0)
        mask_tensor.append(mask_tensor_bin)

    image_tensor = torch.stack(image_tensor)
    mask_tensor = torch.stack(mask_tensor)

    # División entrenamiento / validación
    x_train, x_val, y_train, y_val = train_test_split(image_tensor, mask_tensor, test_size=0.2, random_state=42)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)

    unet = UNet(n_channels=3, n_classes=2).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_iou_list = []
    val_iou_list = []

    for epoch in range(10):
        unet.train()
        running_loss = 0.
        running_iou = 0.

        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = unet(image)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred_class = torch.argmax(pred, dim=1)
            intersection = ((pred_class == 1) & (target == 1)).float().sum((1, 2))
            union = ((pred_class == 1) | (target == 1)).float().sum((1, 2))
            iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
            running_iou += iou

        train_loss_list.append(running_loss / len(train_loader))
        train_iou_list.append(running_iou / len(train_loader))

        # Validación
        unet.eval()
        val_loss = 0.
        val_iou = 0.
        with torch.no_grad():
            for image, target in val_loader:
                image = image.to(device)
                target = target.to(device)

                pred = unet(image)
                loss = criterion(pred, target)
                val_loss += loss.item()

                pred_class = torch.argmax(pred, dim=1)
                intersection = ((pred_class == 1) & (target == 1)).float().sum((1, 2))
                union = ((pred_class == 1) | (target == 1)).float().sum((1, 2))
                iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
                val_iou += iou

        val_loss_list.append(val_loss / len(val_loader))
        val_iou_list.append(val_iou / len(val_loader))

        print(f"Epoch {epoch+1} | Train Loss: {train_loss_list[-1]:.4f} | Val Loss: {val_loss_list[-1]:.4f} | Train IoU: {train_iou_list[-1]:.4f} | Val IoU: {val_iou_list[-1]:.4f}")

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

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou_list, label='Train IoU', marker='o')
    plt.plot(epochs, val_iou_list, label='Validation IoU', marker='o')
    plt.title('Jaccard Index (IoU) por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_segmentacion.png")
    plt.show()