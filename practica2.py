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
import sklearn

if __name__ == '__main__':
    images = os.listdir('./data/Image/')
    print("Total imágenes:", len(images))

    image_tensor = list()
    mask_tensor = list()

    for image in images:
        try:
            dd = PIL.Image.open(f'./data/Image/{image}')
            tt = torchvision.transforms.functional.pil_to_tensor(dd)
            tt = torchvision.transforms.functional.resize(tt, (100, 100))
            tt = tt[None, :, :, :]
            tt = torch.tensor(tt, dtype=torch.float) / 255.

            if tt.shape != (1, 3, 100, 100):
                continue

            mask = image.replace('.jpg', '.png')
            dd = PIL.Image.open(f'./data/Mask/{mask}')
            mm = torchvision.transforms.functional.pil_to_tensor(dd)
            mm = mm.repeat(3, 1, 1)
            mm = torchvision.transforms.functional.resize(mm, (100, 100))
            mm = mm[:1, :, :]

            mm = torch.tensor((mm > 0.).detach().numpy(), dtype=torch.long)
            mm = torch.nn.functional.one_hot(mm)
            mm = torch.permute(mm, (0, 3, 1, 2))
            mm = torch.tensor(mm, dtype=torch.float)

            image_tensor.append(tt)
            mask_tensor.append(mm)

        except Exception as e:
            print(f"Error al procesar {image}: {e}")
            continue

    image_tensor = torch.cat(image_tensor)
    masks_tensor = torch.cat(mask_tensor)

    print("Imagenes shape:", image_tensor.shape)
    print("Máscaras shape:", masks_tensor.shape)

    # División entrenamiento/validación (80/20)
    indices = list(range(len(image_tensor)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_images = image_tensor[train_idx]
    val_images = image_tensor[val_idx]
    train_masks = masks_tensor[train_idx]
    val_masks = masks_tensor[val_idx]

    unet = UNet(n_channels=3, n_classes=2)

    # Dataloaders
    batch_size = 16
    train_loader = zip(torch.utils.data.DataLoader(train_images, batch_size=batch_size),
                       torch.utils.data.DataLoader(train_masks, batch_size=batch_size))
    val_loader = zip(torch.utils.data.DataLoader(val_images, batch_size=batch_size),
                     torch.utils.data.DataLoader(val_masks, batch_size=batch_size))

    # Optimizador y loss
    optim = torch.optim.Adam(unet.parameters(), lr=0.001)
    cross_entropy = torch.nn.CrossEntropyLoss()

    # Métricas
    train_loss_list = []
    val_loss_list = []
    train_iou_list = []
    val_iou_list = []

    for epoch in range(10):
        unet.train()
        train_loss = 0.
        train_iou = []

        for image, target in train_loader:
            optim.zero_grad()
            pred = unet(image)
            loss = cross_entropy(pred, target)
            train_loss += loss.item()
            loss.backward()
            optim.step()

            _, pred_label = torch.max(pred, dim=1)
            _, target_label = torch.max(target, dim=1)
            iou = torch.sum(pred_label == target_label, dim=(1,2)) / 10000.
            train_iou.append(torch.mean(iou).item())

        train_loss_list.append(train_loss)
        train_iou_list.append(sum(train_iou) / len(train_iou))

        # Validación
        unet.eval()
        val_loss = 0.
        val_iou = []
        with torch.no_grad():
            for image, target in val_loader:
                pred = unet(image)
                loss = cross_entropy(pred, target)
                val_loss += loss.item()

                _, pred_label = torch.max(pred, dim=1)
                _, target_label = torch.max(target, dim=1)
                iou = torch.sum(pred_label == target_label, dim=(1,2)) / 10000.
                val_iou.append(torch.mean(iou).item())

        val_loss_list.append(val_loss)
        val_iou_list.append(sum(val_iou) / len(val_iou))

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train IoU: {train_iou_list[-1]:.4f}, Val IoU: {val_iou_list[-1]:.4f}")

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
    plt.show()
