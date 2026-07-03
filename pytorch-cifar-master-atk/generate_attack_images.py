#imports 
import torch
import torchvision
from torch.utils.data import TensorDataset
import torchattacks
from torchvision.transforms.functional import to_tensor, to_pil_image
from datetime import datetime       
import torch.nn as nn
from PIL import Image
from models import *

import os
from torchvision import transforms


def load_cifar10_dataset():
    # Set the paths for dataset and output folders
    dataset_path = os.path.join(os.getcwd(), "data")
    output_path = os.path.join(os.getcwd(), "atk_images/pgd_val")

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define the dataset transformations
    transform = transforms.Compose([
#             transforms.RandomResizedCrop(size= 32, scale = [0.2, 1.0]),
            transforms.ToTensor()
        ])

    # Load the CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform)

    #LOADING THE MODEL
    criterion = nn.CrossEntropyLoss()
    # net_saved = ResNet18()
#     from torchvision import models
#     net_pre_trained = models.resnet18(pretrained=True)
#     # optimizer = optim.SGD(net_pre_trained.parameters(), lr=lr,
#     #                     momentum=0.9, weight_decay=5e-4)

    net = ResNet34()
    checkpoint = torch.load('./Results/resnet34/train_save/resnet34_own_8996.pth')
    


#     checkpoint = torch.load('./checkpoint/resnet18_pretrained')

    net.load_state_dict(checkpoint)

    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available. GPU will be used for training.")
    else:
        device = 'cpu'

    net = net.to(device)
    atk = torchattacks.PGD(net, eps=8/255, alpha=3/255, steps=3)

#     atk = torchattacks.FGSM(net, eps= 12/255)

    # Loop through each image and save as a PIL image
    for i, (image, label) in enumerate(dataset):
        # image , label = image.to(device), label.to(device)
        save_path = os.path.join(output_path, f"{i}.png")
        # Convert target to tensor
        target_tensor = torch.tensor([label])
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)  # Add batch dimension

        # Convert image to tensor
        # image_tensor = to_tensor(image)
        image = image.unsqueeze(0)  # Add batch dimension

        
        # print(image.shape, target_tensor.shape)
        x_adv = atk(image, target_tensor)

        x_adv = x_adv.squeeze(0)
        # Convert adversarial image tensor back to PIL image


        pil_image = transforms.ToPILImage()(x_adv)
#         pil_image = transform_late(pil_image)
        pil_image.save(save_path)

        if (i + 1) % 100 == 0:
            print(f"Saved {i + 1} images\n")

    print("All images saved successfully!")


if __name__ == '__main__':
    load_cifar10_dataset()

