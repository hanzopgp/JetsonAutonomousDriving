import cv2
import os

IMG_FOLDER = "images"
NUM_EACH_POSE = 100
RESOLUTION = (224, 224)

POSES = [
	"PALM",     # palm facing camera
	"FIST",     # fist facing camera
	"FINGER",    # using two fingers to make a V sign
	"LEFT",     # with right arm, making a pistol using index and middle fingers perpendicular to thumb pointing to the left
	"RIGHT",    # invert of LEFT, with left arm
]

def check_img_dir():
	if not os.path.isdir(IMG_FOLDER):
		os.mkdir(IMG_FOLDER)
		print("Created new folder %s in the current directory to save images" % (IMG_FOLDER))
	else:
		print("Folder %s already exists in the current directory. Old files may get replaced by new files" % (IMG_FOLDER))

def get_pose():
	for pose in POSES:
		for i in range(NUM_EACH_POSE):
			yield i+1, pose

def camera_loop(vc):
	cam_ok, frame = vc.read()
	if not cam_ok:
		print("An error occured with the camera")
		return
	
	ip = get_pose()
	i, pose = next(ip)
	print("%s [%d/%d]..." % (pose, i, NUM_EACH_POSE))
	while True:
		cam_ok, image = vc.read()
		if not cam_ok:
			print("An error occured with the camera")
			return
		
		h, w, _ = image.shape
		x_start = (w - h) // 2						# x-coordinate of image to start cropping
		image = image[:, x_start : x_start+h, :]	# crop image to a square
		image = cv2.resize(image, RESOLUTION)
		cv2.imshow("preview", image)
		key = cv2.waitKey(20)
		if key == 27:   # exit on ESC
			return
		elif key == 32:	# spacebar
			cv2.imwrite("%s/%s_%d.jpg" % (IMG_FOLDER, pose, i), image)
			print("  OK")
			try:
				i, pose = next(ip)
				print("%s [%d/%d]..." % (pose, i, NUM_EACH_POSE))
			except:
				print("Finished")
				return

if __name__ == '__main__':
	check_img_dir()
	
	# Check if camera is working
	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(1)
	if vc.isOpened():
		camera_loop(vc)
	else:
		print("Cannot create a new camera window")
	vc.release()
	cv2.destroyWindow("preview")
