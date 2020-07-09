from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Dropout
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

class Styler:
	def __init__(target_autoencoder, art_autodecoder, target_image):

		image_shape = target_image.shape

		target_encoder = target_autoencoder.get_encoder()
		art_decoder = art_autoencoder.get_decoder()

		target_obj = Input(shape = image_shape)
		synthetised_obj = self.target_encoder(target_obj)
		styled_obj = self.art_decoder(synthetised_obj)

		self.Styler = Model(target, styled_obj)
		self.Styler.compile(optimizer = 'adamax', loss = 'mse')

	

class Autoencoder:
	def __init__(self):
		pass

	def build_by_parts(self, encoder, decoder, image_shape):

		target_encoder = target_autoencoder.get_encoder()
		art_decoder = art_autoencoder.get_decoder()

		input_obj = Input(shape = image_shape)
		encoded_obj = self.target_encoder(input_obj)
		reconstructed_obj = self.art_decoder(encoded_obj)

		self.model = Model(target, styled_obj)
		self.model.compile(optimizer = 'adamax', loss = 'mse')

		print(self.model.summary())



	def build_by_layers(self, image_shape, encoding_size):
				
		# The encoder
		encoder = Sequential()
		encoder.add(InputLayer(input_shape = image_shape))
		encoder.add(Conv2D(encoding_size ** 2, encoding_size ** 2, activation='relu', padding="same", input_shape = image_shape))
		encoder.add(Dropout(0.2))
		encoder.add(Flatten())
		encoder.add(Dense(encoding_size ** 2))
		encoder.add(Dropout(0.2))
		encoder.add(Dense(encoding_size))

		# The decoder
		decoder = Sequential()
		decoder.add(InputLayer((encoding_size,)))
		decoder.add(Dense(encoding_size ** 2))
		decoder.add(Dense(np.prod(image_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
		decoder.add(Reshape(image_shape))

		self.encoder = encoder
		self.decoder = decoder

		# Adding all up

		input_obj = Input(shape = image_shape)
		encoded_obj = self.encoder(input_obj)
		reconstructed_obj = self.decoder(encoded_obj)

		self.model = Model(input_obj, reconstructed_obj)
		self.model.compile(optimizer = 'adamax', loss = 'mse')

		print(self.model.summary())

	def train(self, epochs, train_images, test_images):
		
		history = self.model.fit(x = train_images, y = train_images, 
			epochs = epochs,  validation_data = (test_images, test_images))
		
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

		plt.subplot(1,3,1)
		plt.title("Original")
		show_image(image)

		
		plt.subplot(1,3,2)
		plt.title("Encoding")
		
		plt.imshow((encoded_obj.reshape([len(encoded_obj),1]) + 0.5) * 255)
		

		plt.subplot(1,3,3)
		plt.title("Reconstructed")
		show_image(reconstructed_obj)

		plt.show()

		cv2.imwrite('Result.jpg', cv2.cvtColor((reconstructed_obj + 0.5) * 255.0, cv2.COLOR_RGB2BGR))
	
	def get_autoencoder(self):
		return self.autoencoder

	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def save_model(self, name):
		self.autoencoder.save(name)