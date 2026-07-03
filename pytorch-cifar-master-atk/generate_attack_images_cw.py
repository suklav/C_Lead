#imports 
import torch
import torchvision
from torch.utils.data import TensorDataset
import torchattacks
from torchvision.transforms.functional import to_tensor, to_pil_image
from datetime import datetime       
import torch.nn as nn
from PIL import Image

import os
from torchvision import transforms


def load_cifar10_dataset():
    # Set the paths for dataset and output folders
    dataset_path = os.path.join(os.getcwd(), "dataset")
    output_path = os.path.join(os.getcwd(), "atk_images/cw_crop_aug")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define the dataset transformations
    transform = transforms.Compose([
            transforms.RandomResizedCrop(size= 32, scale = [0.2, 1.0]),
            transforms.ToTensor()
        ])

    transform_late = transforms.Compose([
            # transforms.RandomResizedCrop(size= 0.8, scale = [0.2, 1.0]),
            # transforms.CenterCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation = 0.4, hue = 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p = 0.2)
            # transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    # Load the CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)

    #LOADING THE MODEL
    criterion = nn.CrossEntropyLoss()
    # net_saved = ResNet18()
    from torchvision import models
    net_pre_trained = models.resnet18(pretrained=True)
    # optimizer = optim.SGD(net_pre_trained.parameters(), lr=lr,
    #                     momentum=0.9, weight_decay=5e-4)

    checkpoint = torch.load('./Results/resnet34/train_save/resnet34_own_8996.pth')

    net_pre_trained.load_state_dict(checkpoint['net'])

    atk = torchattacks.CW(net_pre_trained, c=1, kappa=0, steps=50, lr=0.01)

    # Loop through each image and save as a PIL image
    for i, (image, label) in enumerate(dataset):
        save_path = os.path.join(output_path, f"{i}.png")
        # Convert target to tensor
        target_tensor = torch.tensor([label])  

        # Convert image to tensor
        # image_tensor = to_tensor(image)
        image = image.unsqueeze(0)  # Add batch dimension

        x_adv = atk(image, target_tensor)

        x_adv = x_adv.squeeze(0)
        # Convert adversarial image tensor back to PIL image


        pil_image = transforms.ToPILImage()(x_adv)
        pil_image = transform_late(pil_image)
        pil_image.save(save_path)

        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1} images\n")

    print("All images saved successfully!")


if __name__ == '__main__':
    load_cifar10_dataset()

