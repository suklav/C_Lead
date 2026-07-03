import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.pylab as plt2
import os

from models import *
import torchattacks


# from resnet50 import ResNet50
# from resnet34 import ResNet34

from torch.optim.lr_scheduler import ReduceLROnPlateau


#Check GPU, connect to it if it is available 
device = ''
if torch.cuda.is_available():
	device = 'cuda'
	print("CUDA is available. GPU will be used for training.")
else:
	device = 'cpu'
	

# Preparing Data
print("==> Prepairing data ...")

transform_validation = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

#Download Train and Validation data and apply transformation
validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_validation)

#Put data into trainloader, specify batch_size
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
model = ResNet([3,4,6,3])
model = ResNet50_x(model)



checkpoint_1 = torch.load('./Results/resnet50/trained/backbone_transfer.pth')
checkpoint_2 = torch.load('./Results/resnet50/trained/linear_transfer.pth')
model.backbone.load_state_dict(checkpoint_1)
model.linear.load_state_dict(checkpoint_2)








model.to(device)
# Print the weights of each layer
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f'Layer: {name}, Shape: {param.data.shape}')
#         print(param.data)



from torch.utils.data import Dataset
from PIL import Image

    
class AdversarialDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = sorted(os.listdir(folder_path))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.folder_path, self.image_files[index])
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image



#Generating model
model_x = ResNet34()
checkpoint_1 = torch.load('./Results/resnet34/train_save/resnet34_own_8996.pth')
model_x.load_state_dict(checkpoint_1)

model_x.to(device)

atk = torchattacks.PGD(model_x, eps=8/255, alpha=3/255, steps=3)

# Testing function

import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def test_adversarial(model, testloader):
    model.eval()
    correct = 0
    total = 0




    # adversarial_dataset = AdversarialDataset('./atk_images/fgsm', transform=ToTensor())
    adversarial_dataset = AdversarialDataset('./../Unsupervised-Classification/atk_images/pgd', transform=ToTensor())

    adversarial_loader = DataLoader(adversarial_dataset, batch_size=testloader.batch_size, shuffle=False)
    # atk = torchattacks.PGD(model, eps=8/255, alpha=3/255, steps=3)
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        
        targets = targets.unsqueeze(1).unsqueeze(2)
        # inputs = inputs.squeeze(0)
        # print(inputs.shape, targets.shape)
        adversarial_images = atk(inputs, targets)

        # # Load the corresponding adversarial images from the folder
        # adversarial_images = next(iter(adversarial_loader))
        # adversarial_images = adversarial_images.to(device)
        outputs = model(adversarial_images)

        _, predicted = outputs.max(1)
        # print(predicted)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print('Accuracy after {} batches: {:.3f}%'.format(batch_idx + 1, accuracy))

    final_accuracy = 100. * correct / total
    print('Final Adversarial Test Accuracy: {:.3f}%'.format(final_accuracy))
#         if (batch_idx + 1) % 10 == 0:
#             print('Accuracy after {} batches: {:.3f}%'.format(batch_idx + 1, 100. * correct / total))

#     print('Final Adversarial Test Accuracy: {:.3f}%'.format(100. * correct / total))

# ...



# Testing the model on adversarial examples
test_adversarial(model, validation_loader)  # Specify the desired epsilon value