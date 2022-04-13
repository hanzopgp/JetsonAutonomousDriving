import cv2
import time
import numpy as np
import os

import torch
from torch import nn
from torchvision import models
from torch.nn.functional import softmax


LABELS = ['FINGER', 'FIST', 'LEFT', 'PALM', 'RIGHT']
INPUT_SIZE = 224
N_CLASS = 5
model_path = "vgg.pth"

model = models.vgg16(pretrained=True)
model.classifier[0] = nn.Linear(25088, 8192)
model.classifier[3] = nn.Linear(8192, 1024)
model.classifier[6] = nn.Linear(1024, N_CLASS)
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
	# img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
	img = img.reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
	img = torch.tensor(img).float()
	# print(img.shape)
	# after = int(time.time() - before)
	with torch.no_grad():
		print(softmax(model(img)), end='\r')
		# print(LABELS[np.argmax(softmax(model(img)))], end='\r')

vid.release()
cv2.destroyAllWindows()