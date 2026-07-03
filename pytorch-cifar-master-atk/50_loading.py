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
	

model = ResNet([3,4,6,3])
model = ResNet50_x(model)


# Load the checkpoint
checkpoint = torch.load('./Results/resnet50/resnet50_9325.pth')

# print(checkpoint)

# Create a new model without the linear layer
# model.backbone = ResNet34_class()

# Load the state_dict of the backbone model from the checkpoint
model.backbone.load_state_dict(checkpoint, strict=False)
model.linear.load_state_dict(checkpoint, strict=False)

# # Print the weights of each layer
# for name, param in backbone.named_parameters():
#     if param.requires_grad:
#         print(f'Layer: {name}, Shape: {param.data.shape}')
#         print(param.data)

# Save the backbone model
torch.save(model.backbone.state_dict(), './Results/resnet50/backbone50.pth')
torch.save(model.linear.state_dict(), './Results/resnet50/linear50.pth')
