import torchvision
import torch
import PIL
from PIL import Image
from unet import UNet
import os
import torch.nn as nn

if __name__ == '__main__':

    images = os.listdir('./data/Image/')
    masks = os.listdir('./data/Mask/')

    print(len(images), len(masks))

    image_tensor = list()
    mask_tensor = list()
    for image in images:
        dd = PIL.Image.open(f'./data/Image/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = torchvision.transforms.functional.resize(tt, (100, 100))

        tt = tt[None, :, :, :]
        tt = torch.tensor(tt, dtype=torch.float) / 255.

        if tt.shape != (1, 3, 100, 100):
            continue

        image_tensor.append(tt)

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

        mask_tensor.append(mm)

    image_tensor = torch.cat(image_tensor)
    print(image_tensor.shape)

    masks_tensor = torch.cat(mask_tensor)
    print(masks_tensor.shape)

    unet = UNet(n_channels=3, n_classes=2)

    dataloader_train_image = torch.utils.data.DataLoader(image_tensor, batch_size=64)
    dataloader_train_target = torch.utils.data.DataLoader(masks_tensor, batch_size=64)

    optim = torch.optim.Adam(unet.parameters(), lr=0.001)
    cross_entropy = torch.nn.CrossEntropyLoss()

    loss_list = list()
    jaccard_list = list()
    for epoch in range(10):
        running_loss = 0.
        unet.train()

        jaccard_epoch = list()
        for image, target in zip(dataloader_train_image, dataloader_train_target):
            
            pred = unet(image)

            loss = cross_entropy(pred, target)
            running_loss += loss.item()

            loss.backward()
            optim.step()

        for image, target in zip(dataloader_train_image, dataloader_train_target):

            pred = unet(image)

            _,pred_unflatten = torch.max(pred, dim = 1)
            _,target_unflatten = torch.max(target, dim = 1)

            intersection = torch.sum(pred_unflatten == target_unflatten, dim=(1,2))/10000.

            jaccard_epoch.append(torch.mean(intersection).detach())

        jaccard_list.append(sum(jaccard_epoch) / len(jaccard_epoch))
        loss_list.append(running_loss)




