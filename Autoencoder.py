from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Dropout
from keras.models import Sequential, Model
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

class Autoencoder:
	def __init__(self):
		pass

	def build_by_parts(self, encoder, decoder, image_shape):

		self.encoder = encoder
		self.decoder = decoder

		input_obj = Input(shape = image_shape)
		encoded_obj = self.encoder(input_obj)
		reconstructed_obj = self.decoder(encoded_obj)

		self.model = Model(input_obj, reconstructed_obj)
		self.model.compile(optimizer = 'adamax', loss = 'mse')

		print(self.model.summary())



	def build_by_layers(self, image_shape, encoding_size, dropout):
				
		# Main layers
		input_layer = InputLayer(image_shape)
		mid_encoding_layer = Dense(encoding_size ** 2)
		final_encoding_layer = Dense(encoding_size)

		encoded_input_layer = InputLayer((encoding_size,))
		mid_decoding_layer = Dense(encoding_size ** 2)
		final_decoding_layer = Dense(np.prod(image_shape))

		#The encoder
		encoder = Sequential()
		encoder.add(input_layer)
		encoder.add(Flatten())
		encoder.add(mid_encoding_layer)
		encoder.add(Dropout(dropout))
		encoder.add(final_encoding_layer)

		# The decoder
		decoder = Sequential()
		decoder.add(encoded_input_layer)
		decoder.add(mid_decoding_layer)
		decoder.add(final_decoding_layer) 
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
		# plt.show()

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
	
	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def save_model(self, name):
		self.autoencoder.save(name)