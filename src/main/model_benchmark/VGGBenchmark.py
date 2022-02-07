import torch
from torch import nn
from torchvision import models, transforms
import torchvision
import os
from time import time

def get_test_data(dataloader, size):
    X_test, Y_test = next(iter(dataloader))
    batch_size = len(X_test)
    n = size//batch_size
    for i, batch in enumerate(dataloader):
        if i < n:
            X_tmp, Y_tmp = batch
            X_test = torch.cat((X_test, X_tmp), 0)
            Y_test = torch.cat((Y_test, Y_tmp), 0)
    return X_test, Y_test

input_size = 224
batch_size = 128

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transformvggTrain=transforms.Compose([ # Cette fois on utilise pas de grayscale car nous avons un gros modele pré-entrainé
        transforms.RandomResizedCrop(input_size), # selection aléatoire d'une zone de la taille voulue (augmentation des données en apprentissage)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transformvggTest=transforms.Compose([
        transforms.Resize(input_size), # selection de la zone centrale de la taille voulue
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

vgg_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformvggTrain)
vgg_trainloader = torch.utils.data.DataLoader(vgg_trainset, batch_size=batch_size, pin_memory=True, shuffle=True)

vgg_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformvggTest)
vgg_testloader = torch.utils.data.DataLoader(vgg_testset, batch_size=batch_size, pin_memory=True, shuffle=True)

X_test, Y_test = get_test_data(vgg_testloader, 1000)

PATH = "models/"
model = models.vgg16(pretrained=True)
model.classifier[0] = nn.Linear(25088, 8192)
model.classifier[3] = nn.Linear(8192, 1024)
model.classifier[6] = nn.Linear(1024, 10)
model.load_state_dict(torch.load(os.path.join(PATH,"vgg.pth"),map_location='cpu'))
model.eval()

for t in (20,40,60,80,100,120):
    t0 = time()
    model(X_test[:t])
    print("FPS:", t, " --> seconds:", (time() - t0))