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

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR


#Check GPU, connect to it if it is available 
device = ''
if torch.cuda.is_available():
	device = 'cuda'
	print("CUDA is available. GPU will be used for training.")
else:
	device = 'cpu'


BEST_ACCURACY = 0

# Preparing Data
print("==> Prepairing data ...")
#Transformation on train data
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	# transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
	])

#transformation on validation data
transform_validation = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

#Download Train and Validation data and apply transformation
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
validation_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_validation)

#Put data into trainloader, specify batch_size
train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=128, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Function to show CIFAR images
def show_data(image):
	plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation='bicubic')
	plt.show()

#show_data(train_data[0])


# Model
model_1 = ResNet34_class()
model = ResNet34_x(model_1)



checkpoint_1 = torch.load('./Results/resnet34/backbone_34_trained.pth')
# checkpoint_2 = torch.load('./Results/resnet34/trained/linear_transfer_11.pth')
model.backbone.load_state_dict(checkpoint_1)
# model.linear.load_state_dict(checkpoint_2)



for name, param in model.backbone.named_parameters():
    param.requires_grad = False


#model = CNN_batch()
#Pass model to GPU
model = model.to(device)
model.train()

# for name, param in model.named_parameters():
# #     if param.requires_grad:
#     print(f'Layer: {name}, Shape: {param.data.shape}')
#     print(param.data)
    


optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
# scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

# Define the scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)



length_train = len(train_data)
length_validation = len(validation_data)
#print(length_train)
#print(len(train_loader))
num_classes = 10



# Testing
from torch.utils.data import Dataset
from PIL import Image
import torchattacks
    
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
    

# atk = torchattacks.PGD(model_x, eps=8/255, alpha=3/255, steps=3)

#Generating model
# from torchvision import models
# model_x = ResNet18
attack_percentage = 0.5
model_x = ResNet34()
checkpoint_1 = torch.load('./Results/resnet34/train_save/resnet34_own_8996.pth')
model_x.load_state_dict(checkpoint_1)

model_x.to(device)
# atk = torchattacks.PGD(model_x, eps=4/255, alpha=3/255, steps=3)
atk = torchattacks.FGSM(model_x, eps=8/255)

# atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
# (0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
# atk = torchattacks.CW(model_x, c=1, kappa=0, steps=50, lr=0.01)
# atk = torchattacks.EOTPGD(model, eps=8/255, alpha=3/255, steps=3, eot_iter=2)



# Testing function

import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def test_adversarial(model, testloader):
    model.eval()
    correct = 0
    total = 0


    # atk = torchattacks.PGD(model_x, eps=15/255, alpha=3/255, steps=3)


    # atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # adversarial_dataset = AdversarialDataset('./atk_images/fgsm', transform=ToTensor())
    # # adversarial_dataset = AdversarialDataset('./../Unsupervised-Classification/atk_images/fgsm', transform=ToTensor())

    # adversarial_loader = DataLoader(adversarial_dataset, batch_size=testloader.batch_size, shuffle=False)
    # # atk = torchattacks.PGD(model, eps=8/255, alpha=3/255, steps=3)
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # inputs, targets = inputs.to(device), targets.to(device)

        
        targets = targets.unsqueeze(1).unsqueeze(2)
        # inputs = inputs.squeeze(0)
        # print(inputs.shape, targets.shape)
        # inputs, targets = inputs.to(device), targets.to(device)
        
        # print(inputs.shape, targets.shape)
        adversarial_images = atk(inputs, targets)

        # # Load the corresponding adversarial images from the folder
        # adversarial_images = next(iter(adversarial_loader))
        # adversarial_images = adversarial_images.to(device)
        outputs = model(adversarial_images)

        _, predicted = outputs.max(1)
        # print(predicted)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        correct += predicted.eq(targets.to(predicted.device)).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print('Accuracy after {} batches: {:.3f}%'.format(batch_idx + 1, accuracy))

    final_accuracy = 100. * correct / total
    print('Final Adversarial Test Accuracy: {:.3f}%'.format(final_accuracy))









#Training
def train(epochs):
	global BEST_ACCURACY
	dict = {'Train Loss':[], 'Train Acc':[], 'Validation Loss':[], 'Validation Acc':[]}
	for epoch in range(epochs):
		print("\nEpoch:", epoch+1, "/", epochs)
		cost = 0
		correct = 0
		total = 0
		woha = 0

		for i, (x,y) in enumerate(train_loader):
            
			woha += 1
			model.train()
			# if(epoch+1) % 3 == 0:
			# z = y.unsqueeze(1).unsqueeze(2)
			# x = atk(x, z)

			# Check if the current batch should be attacked
			if torch.rand(1) < attack_percentage:
				# atk = torchattacks.FGSM(model, eps=8/255)
				z = y.unsqueeze(1).unsqueeze(2)
				x = atk(x, z)

			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			yhat = model(x)
			yhat = yhat.reshape(-1, 10)
			loss = criterion(yhat, y)
			loss.backward()
			optimizer.step()
			
			cost += loss.item()

			_, yhat2 = torch.max(yhat.data, 1)
			correct += (yhat2 == y).sum().item()
			total += y.size(0)
			# print("\nAcc:", correct/total, correct, "/", total)

		my_loss = cost/len(train_loader)
		my_accuracy = 100*correct/length_train

		dict['Train Loss'].append(my_loss)
		dict['Train Acc'].append(my_accuracy)

		print('Tain Loss:', my_loss)
		print('Train Accuracy:', my_accuracy,'%')


		cost = 0
		correct = 0

		with torch.no_grad():
			for x, y in validation_loader:
				x, y = x.to(device), y.to(device)
				model.eval()
				yhat = model(x)
				yhat = yhat.reshape(-1, 10)
				loss = criterion(yhat, y)
				cost += loss.item()
				
				_, yhat2 = torch.max(yhat.data, 1)
				correct += (yhat2 == y).sum().item()

		my_loss = cost/len(validation_loader)
		my_accuracy = 100*correct/length_validation

		dict['Validation Loss'].append(my_loss)
		dict['Validation Acc'].append(my_accuracy)

		print('Validation Loss:', my_loss)
		print('Validation Accuracy:', my_accuracy,'%')
        
		# Update the optimizer parameters
		optimizer.step()

		# Update the scheduler based on the validation accuracy
		# scheduler.step(my_accuracy)
		if (epoch + 1) % 15 == 0:
				for param_group in optimizer.param_groups:
					param_group['lr'] *= 0.5

		# Zero the gradients
		optimizer.zero_grad()

		if (epoch+1) % 10 == 0:
			# Calling the testing function
			test_adversarial(model, validation_loader)

		#Save the model if you get best accuracy on validation data
		if my_accuracy > BEST_ACCURACY:
			BEST_ACCURACY = my_accuracy
			print('Saving the model ...')
			model.eval()
			if not os.path.isdir('checkpoint'):
			    os.mkdir('checkpoint')
			torch.save(model.state_dict(), './checkpoint/trained/resnet50.pth')

	print("TRAINING IS FINISHED !!!")
	return dict

#Start training
results = train(50)


torch.save(model.backbone.state_dict(), './Results/resnet50/trained/backbone_transfer.pth')
torch.save(model.linear.state_dict(), './Results/resnet50/trained/linear_transfer.pth')