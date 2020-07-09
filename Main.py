from Autoencoder import Autoencoder
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(directory, resize_shape = None):
	if os.path.exists(directory):
		pass	
	else:
		print('Directory ', dir_path, ' does not exist')

	files = glob.glob(directory + '/*')

	images =[]

	if resize_shape:
		height, width = resize_shape
		print('Resizing images to ',height, 'x', width)

	for file in files:
		image = cv2.imread(file).astype('float32') / 255.0

		if resize_shape == None:
			images.append(image)
		else:
			image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
			images.append(image)

	images = np.asarray(images)
	print(len(images), ' images loaded from ', directory)

	return images

def split_data(data, ratio):
	n = int(len(data) * ratio) + 1
	return data[:n], data[n:]

def show_image(x):
    plt.imshow(np.clip(x, 0, 1))

shape = (150,150)

art_data = load_images('Art_Data', shape)
target_data = load_images('Target_Data', shape)


cv2.imshow('Art',art_data[0])
cv2.waitKey(0)

art_autoen = Autoencoder(art_data, art_data, 2500)
art_autoen.train(epochs = 350)
art_autoen.visualize(art_data[0])

	

