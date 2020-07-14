#https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f
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
		#cv2 uses BGR colour scheme, matplotlib uses RGB
		image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
		image = image.astype('float32') / 255 - 0.5

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
    plt.imshow(np.clip(x + 0.5, 0, 1))


shape = (200,200)
epochs = 50

k = 2
encoding_size = int(k * np.log(shape[0] * shape[1]))
print("Encoding size: ",encoding_size)

art_data = load_images('Art_Data', shape)
target_data = load_images('Target_Data', shape)

white_canvas = np.ones(target_data[0].shape)

print('Encoding Size: ', encoding_size)

print('Training Art Autoencoder')
art_autoen = Autoencoder()
art_autoen.build_by_layers(image_shape = art_data[0].shape, encoding_size = encoding_size)
art_autoen.train(train_images = art_data, test_images = art_data, epochs = epochs)

art_autoen.visualize(art_data[0])

print('Training Target Autoencoder')
target_autoen = Autoencoder()
target_autoen.build_by_layers(image_shape = target_data[0].shape, encoding_size = encoding_size)
target_autoen.train(train_images = target_data, test_images = target_data, epochs = epochs)

target_autoen.visualize(target_data[0])

art_conv, art_encoder, art_decoder, art_deconv = art_autoen.get_blocks()
target_conv, target_encoder, target_decoder, target_deconv = target_autoen.get_blocks()

arthur = Autoencoder()
arthur.build_by_parts(target_conv,target_encoder,art_decoder,art_deconv,target_data[0].shape)
arthur.train(train_images = target_data, test_images = target_data, epochs = int(epochs/2))

arthur.visualize(target_data[0])

#target_autoen.save_model('Autoen')


	

