from PIL import Image
import glob
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
# print(tensorflow. __version__)

def get_images():
    image_list = []
    for filename in glob.glob('labelling_img/*.jpg'): 
        im=Image.open(filename)
        image_list.append(np.array(im))
    return image_list

if __name__ == '__main__':

	X = get_images()

	plt.imshow(X[80])
	print(len(X))

	# datagen = ImageDataGenerator()
	# it = datagen.flow(X, y)
