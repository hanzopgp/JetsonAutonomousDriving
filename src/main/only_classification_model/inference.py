import cv2
import time
import numpy as np
import os

import torch
from torch import nn
from torchvision import models, transforms
from torch.nn.functional import softmax


LABELS = ['FINGER', 'FIST', 'LEFT', 'PALM', 'RIGHT']
INPUT_SIZE = 224
N_CLASS = 5

model_path = "models/best_model.pth"

# training whole vgg classifier + latest feature layers :

# training only some of the classifier layers :

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# model = models.vgg16(pretrained=True)
# model.classifier[0] = nn.Linear(25088, 8192)
# model.classifier[3] = nn.Linear(8192, 1024)
# model.classifier[6] = nn.Linear(1024, N_CLASS)
# model.load_state_dict(torch.load(os.path.join(model_path), map_location='cpu'))
# model.eval() 

model = models.vgg16(pretrained=True)
model.classifier = nn.Sequential(nn.Linear(25088, 100),
                               nn.ReLU(), 
                               nn.Dropout(0.45),        
                               nn.Linear(100, N_CLASS), 
                               nn.Softmax(dim=1)) 
model.load_state_dict(torch.load(os.path.join(model_path), map_location='cpu'))
model.eval() 

cv2.destroyAllWindows()
# before = time.time()
vid = cv2.VideoCapture(1)
while True:

	ret, frame = vid.read()
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
	# img = img.reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
	# img = torch.tensor(img).float()

	transformInference=transforms.Compose([
		transforms.ToPILImage(),
        # transforms.CenterCrop(INPUT_SIZE),
		transforms.Resize(size=(INPUT_SIZE, INPUT_SIZE)),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std),
    ])

	img = transformInference(frame)
	img = img.unsqueeze(0)

	# print(img)

	# print(img.shape)
	# after = int(time.time() - before)
	with torch.no_grad():
		# print(softmax(model(img)), end='\r')
		print(LABELS[np.argmax(softmax(model(img)))], end='\r')
		# print(np.where(model(img).numpy() > 0.8, 1, 0))
		# print(model(img).numpy())
vid.release()
cv2.destroyAllWindows()