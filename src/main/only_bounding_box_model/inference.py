import socket
import cv2
import pickle
import struct
import os

import torch
from torch import nn
from torchvision import models, transforms

INPUT_SIZE = 400
N_CLASS = 4

# load model
model_path = "models/boundingbox_vgg_last.pth"
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
model.load_state_dict(torch.load(os.path.join(model_path), map_location=torch.device('cuda:0')))
model.eval()
transformInference = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize(size=(INPUT_SIZE, INPUT_SIZE)),
	transforms.ToTensor()
])

# start server
# HOST='0.0.0.0'
# PORT=8000
# s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# s.bind((HOST,PORT))
# s.listen(10)

# print("Server ready...")
# conn, addr = s.accept()

# data = b""
# payload_size = struct.calcsize("L") 

cv2.namedWindow("frame")
vc = cv2.VideoCapture(0)

while True:
	# receive stream
	# while len(data) < payload_size:
	# 	data += conn.recv(4096)
	# packed_msg_size = data[:payload_size]
	# data = data[payload_size:]
	# msg_size = struct.unpack("L", packed_msg_size)[0]
	# while len(data) < msg_size:
	# 	data += conn.recv(4096)
	# frame_data = data[:msg_size]
	# data = data[msg_size:]
	# img = pickle.loads(frame_data)

	cam_ok, img = vc.read()
	img = cv2.flip(img, 1)
	img = transformInference(img)
	img = img.unsqueeze(0)
	with torch.no_grad():
		preds = model(img)
		startX, startY, endX, endY = preds.squeeze(0)
		startX, startY, endX, endY = startX.item(), startY.item(), endX.item(), endY.item()
		startX = int(startX * INPUT_SIZE)
		startY = int(startY * INPUT_SIZE)
		endX = int(endX * INPUT_SIZE)
		endY = int(endY * INPUT_SIZE)
		img = img.squeeze(0).permute(1,2,0).numpy().copy()
		# img = img.numpy().copy()
		cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.imshow('frame', img)
	

		# show image
		# cv2.imshow(f"Webcam",img)
		key = cv2.waitKey(1)
		if key == 27:
			cv2.destroyAllWindows()
			vc.release()
			# s.shutdown(socket.SHUT_RDWR)
			break

# cv2.destroyWindow("preview")