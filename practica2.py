import torchvision
import torch
from PIL import Image
from unet import UNet
import os
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Usando dispositivo: {device}')

    images = os.listdir('./data/Image/')
    masks = os.listdir('./data/Mask/')

    print(len(images), len(masks))

    image_tensor = []
    mask_tensor = []

    for image in images:
        img = Image.open(f'./data/Image/{image}')
        img_tensor = torchvision.transforms.functional.pil_to_tensor(img)
        img_tensor = torchvision.transforms.functional.resize(img_tensor, (100, 100))
        img_tensor = img_tensor.clone().detach().float() / 255.

        if img_tensor.shape != (3, 100, 100):
            continue

        image_tensor.append(img_tensor.unsqueeze(0))

        # Cargar y preparar la máscara
        mask_name = image.replace('.jpg', '.png')
        mask = Image.open(f'./data/Mask/{mask_name}')
        mask_tensor_raw = torchvision.transforms.functional.pil_to_tensor(mask)
        mask_tensor_raw = torchvision.transforms.functional.resize(mask_tensor_raw, (100, 100))
        mask_tensor_raw = mask_tensor_raw[:1, :, :]  # Tomamos un canal

        # Binarizamos y pasamos a one-hot
        mask_bin = (mask_tensor_raw > 0).long()
        mask_one_hot = torch.nn.functional.one_hot(mask_bin[0], num_classes=2).permute(2, 0, 1)
        mask_tensor.append(mask_one_hot.unsqueeze(0).float())

    image_tensor = torch.cat(image_tensor)
    masks_tensor = torch.cat(mask_tensor)

    print("Imágenes:", image_tensor.shape)
    print("Máscaras:", masks_tensor.shape)

    # Crear DataLoaders
    dataset = torch.utils.data.TensorDataset(image_tensor, masks_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Modelo
    unet = UNet(n_channels=3, n_classes=2).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Métricas
    train_loss_list = []
    train_iou_list = []

    for epoch in range(10):
        unet.train()
        running_loss = 0.0

        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = unet(imgs)

            # Para CrossEntropyLoss: necesita (N, C, H, W) y (N, H, W)
            loss = loss_fn(outputs, torch.argmax(masks, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluación del IoU
        unet.eval()
        iou_epoch = []
        with torch.no_grad():
            for imgs, masks in dataloader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                outputs = unet(imgs)
                preds = torch.argmax(outputs, dim=1)
                targets = torch.argmax(masks, dim=1)

                intersection = ((preds == 1) & (targets == 1)).sum(dim=(1, 2))
                union = ((preds == 1) | (targets == 1)).sum(dim=(1, 2))
                iou = (intersection.float() / (union.float() + 1e-6)).mean().item()
                iou_epoch.append(iou)

        avg_iou = sum(iou_epoch) / len(iou_epoch)
        train_loss_list.append(running_loss)
        train_iou_list.append(avg_iou)

        print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | IoU: {avg_iou:.4f}")

    # Gráficas
    epochs = list(range(1, 11))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss', marker='o')
    plt.title('Loss por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_iou_list, label='Train IoU', marker='o')
    plt.title('Jaccard Index (IoU) por Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
