import cv2
import time
import numpy as np
import os

import torch
from torch import nn
from torchvision import models, transforms
from torch.nn.functional import softmax

INPUT_SIZE = 400
N_CLASS = 4

model_path = "models/boundingbox_vgg_last.pth"

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# model = models.vgg16(pretrained=True)
# model.classifier[0] = nn.Linear(25088, 8192)
# model.classifier[3] = nn.Linear(8192, 1024)
# model.classifier[6] = nn.Linear(1024, N_CLASS)
# model.load_state_dict(torch.load(os.path.join(model_path), map_location='cpu'))
# model.eval() 

model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(25088, 4096), 
                               nn.ReLU(), 
                            #    nn.Dropout(0.5),        
                               nn.Linear(4096, 1024), 
                               nn.ReLU(), 
#                                nn.Dropout(0.5),        
                               nn.Linear(1024, 256),
                               nn.ReLU(), 
#                                nn.Dropout(0.5),        
                               nn.Linear(256, N_CLASS),
                               nn.Sigmoid())
model.load_state_dict(torch.load(os.path.join(model_path), map_location='cpu'))
model.eval() 

cv2.destroyAllWindows()
vid = cv2.VideoCapture(1)
vid.set(cv2.CAP_PROP_FPS, 5)

while True:

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	transformInference=transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize(size=(INPUT_SIZE, INPUT_SIZE)),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std),
    ])
	ret, frame = vid.read()
	img = transformInference(frame)
	img = img.unsqueeze(0)
	with torch.no_grad():
		preds = model(img)
		startX, startY, endX, endY = preds.squeeze(0)
		startX, startY, endX, endY = startX.item(), startY.item(), endX.item(), endY.item()
		startX = int(startX * INPUT_SIZE)
		startY = int(startY * INPUT_SIZE)
		endX = int(endX * INPUT_SIZE)
		endY = int(endY * INPUT_SIZE)
		img = img.squeeze(0).permute(1,2,0)
		img = img.numpy().copy()
		cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.imshow('frame', img)

vid.release()
cv2.destroyAllWindows()