# '''Train CIFAR10 with PyTorch.'''
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

# import torchvision
# import torchvision.transforms as transforms

# import os
# import argparse

# from models import *
# from models import MyResnet

# from utils import progress_bar
# from torchvision import models


# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # device = 'cpu'
# print(device)
# best_acc = 0  # best test accuracy
# start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# # Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=128, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# # Model

# print('==> Building model..')
# # net = VGG('VGG19')
# # net = ResNet18()
# net = ResNetCifar10(num_classes= 10)
# # net = ResNet34Cifar10(num_classes = 10)
# # net = ResNet50Cifar10(num_classes = 10)

# # net = PreActResNet18()
# # net = GoogLeNet()
# # net = DenseNet121()
# # net = ResNeXt29_2x64d()
# # net = MobileNet()
# # net = MobileNetV2()
# # net = DPN92()
# # net = ShuffleNetG2()
# # net = SENet18()
# # net = ShuffleNetV2(1)
# # net = EfficientNetB0()
# # net = RegNetX_200MF()
# # net = SimpleDLA()

# # for name, para in net.named_parameters():
# #     if para.requires_grad and 'linear' not in name:
# #         para.requires_grad = False
# #     # print("-"*20)
# #     # print(f"name: {name}")
# #     # print("values: ")
# #     # print(para)


# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# if args.resume:
#     # Load checkpoint.
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load('./checkpoint/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']

# from torch.optim.lr_scheduler import ReduceLROnPlateau
# optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)


# # Training
# def train(epoch):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                      % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc


# for epoch in range(start_epoch, start_epoch+300):
#     train(epoch)
#     test(epoch)
#     scheduler.step()
#     #torch.save(net.module.backbone.state_dict(), './Results/resnet18/backbone_pretrained.pth')
#     #torch.save(net.module.linear.state_dict(), './Results/resnet18/linear_pretrained.pth') 
#     #print("Save ckpt done.")
    
# torch.save(net.module.backbone.state_dict(), './Results/resnet18/backbone_pretrained.pth')
# torch.save(net.module.linear.state_dict(), './Results/resnet18/linear_pretrained.pth') 



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


BEST_ACCURACY = 0

# Preparing Data
print("==> Prepairing data ...")
#Transformation on train data
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
	])

#transformation on validation data
transform_validation = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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


#Need to import model a model
# model = ResNet50()
model = ResNetCifar10(num_classes=10)
# model = ResNet34Cifar10(num_classes = 10)
# model = ResNet34()
#model = CNN_batch()
#Pass model to GPU
model = model.to(device)
model.train()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)



length_train = len(train_data)
length_validation = len(validation_data)
#print(length_train)
#print(len(train_loader))
num_classes = 10



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
		scheduler.step(my_accuracy)

		# Zero the gradients
		optimizer.zero_grad()

		#Save the model if you get best accuracy on validation data
		if my_accuracy > BEST_ACCURACY:
			BEST_ACCURACY = my_accuracy
			print('Saving the model ...')
			model.eval()
			if not os.path.isdir('checkpoint'):
			    os.mkdir('checkpoint')
			torch.save(model.state_dict(), './checkpoint/resnet18_own.pth')

	print("TRAINING IS FINISHED !!!")
	return dict



#Start training
results = train(70)


# plt.figure(1)
# plt.plot(results['Train Loss'], 'b', label = 'training loss')
# plt.plot(results['Validation Loss'], 'r', label = 'validation loss')
# plt.title("LOSS")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(['training set', 'validation set'], loc='center right')
# plt.savefig('Loss_ResNet50.png', dpi=300, bbox_inches='tight')

# plt.figure(2)
# plt.plot(results['Train Acc'], 'b', label = 'training accuracy')
# plt.plot(results['Validation Acc'], 'r', label = 'validation accuracy')
# plt.title("ACCURACY")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(['training set', 'validation set'], loc='center right')
# plt.savefig('Accuracy_ResNet50.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

"""
axs[0].plot(results['Train Loss'], 'b', label = 'training loss')
axs[0].plot(results['Validation Loss'], 'r', label = 'validation loss')
axs[0].set_title("LOSS")
axs[0].set(xlabel="Epochs", ylabel="Loss")

axs[1].plot(results['Train Acc'], 'b', label = 'training accuracy')
axs[1].plot(results['Validation Acc'], 'r', label = 'validation accuracy')
axs[1].set_title("ACCURACY")
axs[1].set(xlabel="Epochs", ylabel="Accuracy")

fig.tight_layout()
plt.legend()
plt.show()
"""
torch.save(model.backbone.state_dict(), './Results/resnet18/backbone_own.pth')
torch.save(model.linear.state_dict(), './Results/resnet18/linear_own.pth')





