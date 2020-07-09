from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_image(x):
    plt.imshow(np.clip(x, 0, 1))

class Autoencoder:
	def __init__(self, train_images, test_images, encoding_size):

		self.train_images = train_images
		self.test_images = test_images

		image_shape = train_images[0].shape


		print('Input shape: ', image_shape)

		# The encoder
		encoder = Sequential()
		encoder.add(InputLayer(input_shape = image_shape))
		encoder.add(Flatten())
		encoder.add(Dense(encoding_size))

		# The decoder
		decoder = Sequential()
		decoder.add(InputLayer((encoding_size,)))
		decoder.add(Dense(np.prod(image_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
		decoder.add(Reshape(image_shape))

		self.encoder = encoder
		self.decoder = decoder

		# Adding all up

		input_obj = Input(shape = image_shape)
		encoded_obj = self.encoder(input_obj)
		reconstructed_obj = self.decoder(encoded_obj)

		self.autoencoder = Model(input_obj, reconstructed_obj)
		self.autoencoder.compile(optimizer = 'adamax', loss = 'mse')

		print(self.autoencoder.summary())

	def train(self, epochs):
		history = self.autoencoder.fit(x=self.train_images, y=self.train_images, 
			epochs = epochs,  validation_data = (self.test_images, self.test_images))
		
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper right')
		plt.show()

	def visualize(self, image):

		encoded_obj = self.encoder.predict(image[None])[0]
		reconstructed_obj = self.decoder.predict(encoded_obj[None])[0]

		
		cv2.imshow('Original',image)
		cv2.waitKey(0)

		cv2.imshow('Reconstructed',reconstructed_obj)
		cv2.waitKey(0)
		

		plt.subplot(1,2,1)
		plt.title("Original")
		show_image(image)

		'''
		plt.subplot(1,3,2)
		plt.title("Encoding")
		plt.imshow(encoded_obj.reshape([encoded_obj[-1]//2,-1]))
		'''
		plt.subplot(1,2,2)
		plt.title("Reconstructed")
		show_image(reconstructed_obj)

		plt.show()

	def save_model(name):
		model.save(name)