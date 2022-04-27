import cv2
import os
from random import randrange

IMG_FOLDER = "image"	# subfolder of current working directory to store images
NUM_PER_SZ = 40			# number of images for each bounding-box size
IMG_SIZE = 400			# size of square image

INIT_SIZE = 50
SIZE_STEP = 20


def check_img_dir():
	if not os.path.isdir(IMG_FOLDER):
		os.mkdir(IMG_FOLDER)
		print(f"Created new folder {IMG_FOLDER} in the current directory to save images")
	else:
		print(f"Folder {IMG_FOLDER} already exists in the current directory. Old files may get replaced by new files")

def generate_box():
	i = 0
	for size in range(INIT_SIZE, IMG_SIZE//2+1, SIZE_STEP):
		for _ in range(NUM_PER_SZ):
			i += 1
			box_x, box_y = randrange(IMG_SIZE - size), randrange(IMG_SIZE - size)
			yield i, box_x, box_y, box_x + size, box_y + size

def take_picture(event, x, y, flags, params):
	frame_copy, gen_box = params
	if event == cv2.EVENT_LBUTTONDBLCLK:
		i, box_x, box_y, end_x, end_y = next(gen_box)
		while True:
			cv2.rectangle(frame_copy, (box_x,box_y), (end_x,end_y), (0,255,0), 2)
			cv2.imshow("preview", frame_copy)
			if event == cv2.EVENT_LBUTTONDBLCLK:
				cv2.imwrite(f"{IMG_FOLDER}/{box_x}_{box_y}_{end_x}_{end_y}_{i}.jpg", frame_copy)
				print("\033[92mOK\033[0m")
				print(f"{i} / {(IMG_SIZE // SIZE_STEP) * NUM_PER_SZ}")

def camera_loop(vc):
	cam_ok, frame = vc.read()
	if not cam_ok:
		print("An error occured with the camera")
		return

	total_imgs = (IMG_SIZE // SIZE_STEP) * NUM_PER_SZ
	gen_box = generate_box()
	i, box_x, box_y, end_x, end_y = next(gen_box)
	print(f"{i} / {total_imgs}")

	h, w, _ = frame.shape
	print(h,w)
	img_y_start = (w - h) // 2

	while True:
		cam_ok, frame = vc.read()
		if not cam_ok:
			print("An error occured with the camera")
			return
		frame = cv2.flip(cv2.resize(frame[:,img_y_start:img_y_start + h], (IMG_SIZE, IMG_SIZE)), 1)
		frame_copy = frame.copy()
		# cv2.rectangle(frame, (box_x,box_y), (end_x,end_y), (0,255,0), 2)
		# cv2.imshow("preview", frame)

		cv2.setMouseCallback("preview", take_picture, [frame_copy, gen_box])

		key = cv2.waitKey(20)
		if key == 27:   # exit on ESC
			return
		# elif key == 32:	# spacebar
		# 	cv2.imwrite(f"{IMG_FOLDER}/{box_x}_{box_y}_{end_x}_{end_y}_{i}.jpg", frame_copy)
		# 	print("\033[92mOK\033[0m")
		# 	try:
		# 		i, box_x, box_y, end_x, end_y = next(gen_box)
		# 	except StopIteration:
		# 		print("Finished")
		# 		return

if __name__ == '__main__':
	check_img_dir()
	
	# Check if camera is working
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)
	if vc.isOpened():
		camera_loop(vc)
	else:
		print("Cannot create a new camera window")
	vc.release()
	cv2.destroyWindow("preview")
