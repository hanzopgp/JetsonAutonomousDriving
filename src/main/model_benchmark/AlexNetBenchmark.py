import torch
from torch import nn
from torchvision import models, transforms
import torchvision
import os
from time import time
import onnx
import onnxruntime
from onnx import numpy_helper
import numpy as np

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

transformAlexTrain=transforms.Compose([ # Cette fois on utilise pas de grayscale car nous avons un gros modele pré-entrainé
        transforms.RandomResizedCrop(input_size), # selection aléatoire d'une zone de la taille voulue (augmentation des données en apprentissage)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transformAlexTest=transforms.Compose([
        transforms.Resize(input_size), # selection de la zone centrale de la taille voulue
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

alex_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformAlexTrain)
alex_trainloader = torch.utils.data.DataLoader(alex_trainset, batch_size=batch_size, pin_memory=True, shuffle=True)

alex_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformAlexTest)
alex_testloader = torch.utils.data.DataLoader(alex_testset, batch_size=batch_size, pin_memory=True, shuffle=True)

X_test, Y_test = get_test_data(alex_testloader, 1000)

PATH = "models/"
model = models.AlexNet()
model.classifier[1] = nn.Linear(9216, 2048)
model.classifier[4] = nn.Linear(2048, 1024)
model.classifier[6] = nn.Linear(1024, 10)
model.load_state_dict(torch.load(os.path.join(PATH,"alexnet.pth")))
model.eval()
# for t in (20,40,60,80,100,120):
#     t0 = time()
#     model(X_test[:t])
#     print("FPS:", t, " --> seconds:", (time() - t0))

test_size = 100
dummy_input = torch.randn(test_size, 3, input_size, input_size)  
torch.onnx.export(model,   
                  dummy_input, 
                  str(PATH+"alexnet.onnx"),
                  export_params=True,
                  do_constant_folding=True, 
                  input_names = ['modelInput'],
                  output_names = ['modelOutput'])



X_test = X_test[:test_size]

t0 = time()
pred = model(X_test)
print("Time for ",test_size," images without ONNX inference", (time() - t0))
print(pred.shape)

sess = onnxruntime.InferenceSession(str(PATH+"alexnet.onnx"))
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
t0 = time()
pred = sess.run([output_name], {input_name: np.array(X_test).astype(np.float32)})[0]
print("Time for ",test_size," images with ONNX inference", (time() - t0))
print(pred.shape)