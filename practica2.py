
import torchvision
import torch
import PIL
from PIL import Image
from unet import UNet
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Usando dispositivo: {device}')

    images = os.listdir('./data/Image/')
    masks = os.listdir('./data/Mask/')

    print(len(images), len(masks))

    image_tensor = list()
    mask_tensor = list()
    for image in images:
        dd = PIL.Image.open(f'./data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (128, 128))
        tt = tt.clone().detach().float() / 255.
        if tt.shape != (3, 128, 128):
            continue
        image_tensor.append(tt.unsqueeze(0))

        mask = image.replace('.jpg', '.png')
        dd = PIL.Image.open(f'./data/Mask/{mask}')
        mm = torchvision.transforms.functional.pil_to_tensor(dd)
        mm = torchvision.transforms.functional.resize(mm, (128, 128))
        mm = (mm[:1] > 0).long()
        mm = torch.nn.functional.one_hot(mm[0], num_classes=2).permute(2, 0, 1)
        mask_tensor.append(mm.unsqueeze(0).float())

    image_tensor = torch.cat(image_tensor)
    masks_tensor = torch.cat(mask_tensor)
    print(image_tensor.shape)
    print(masks_tensor.shape)

    # ✅ División manual en entrenamiento y validación (80/20)
    dataset_size = image_tensor.shape[0]
    split = int(0.8 * dataset_size)
    indices = torch.randperm(dataset_size)

    train_indices = indices[:split]
    val_indices = indices[split:]

    train_imgs = image_tensor[train_indices]
    train_masks = masks_tensor[train_indices]
    val_imgs = image_tensor[val_indices]
    val_masks = masks_tensor[val_indices]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_imgs, train_masks), batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_imgs, val_masks), batch_size=64, shuffle=False)

    unet = UNet(n_channels=3, n_classes=2).to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=0.001)
    cross_entropy = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []
    train_iou_list = []
    val_iou_list = []

    for epoch in range(30):
        running_loss = 0.
        unet.train()

        for image, target in train_loader:
            image = image.to(device)
            target = target.to(device)

            pred = unet(image)
            loss = cross_entropy(pred, torch.argmax(target, dim=1))
            running_loss += loss.item()

            optim.zero_grad()
            loss.backward()
            optim.step()

        # IoU entrenamiento
        unet.eval()
        train_iou = []
        with torch.no_grad():
            for image, target in train_loader:
                image = image.to(device)
                target = target.to(device)
                pred = unet(image)
                pred_labels = torch.argmax(pred, dim=1)
                target_labels = torch.argmax(target, dim=1)
                intersection = ((pred_labels == 1) & (target_labels == 1)).sum(dim=(1, 2))
                union = ((pred_labels == 1) | (target_labels == 1)).sum(dim=(1, 2))
                iou = (intersection.float() / (union.float() + 1e-6)).mean().item()
                train_iou.append(iou)

        avg_train_iou = sum(train_iou) / len(train_iou)
        train_loss_list.append(running_loss)
        train_iou_list.append(avg_train_iou)

        # Validación
        val_loss = 0.
        val_iou = []
        with torch.no_grad():
            for image, target in val_loader:
                image = image.to(device)
                target = target.to(device)
                pred = unet(image)
                val_loss += cross_entropy(pred, torch.argmax(target, dim=1)).item()
                pred_labels = torch.argmax(pred, dim=1)
                target_labels = torch.argmax(target, dim=1)
                intersection = ((pred_labels == 1) & (target_labels == 1)).sum(dim=(1, 2))
                union = ((pred_labels == 1) | (target_labels == 1)).sum(dim=(1, 2))
                iou = (intersection.float() / (union.float() + 1e-6)).mean().item()
                val_iou.append(iou)

        avg_val_iou = sum(val_iou) / len(val_iou)
        val_loss_list.append(val_loss)
        val_iou_list.append(avg_val_iou)

        print(f"Epoch {epoch+1} | Train Loss: {running_loss:.4f} | Train IoU: {avg_train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

    # Gráficas
    epochs = list(range(1, len(train_loss_list) + 1))

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
    plt.savefig("resultados_entrenamiento.png")
    plt.show()