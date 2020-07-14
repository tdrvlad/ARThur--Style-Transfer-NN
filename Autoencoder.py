from keras.layers import InputLayer, Conv2D, Conv2DTranspose, Dense, MaxPooling2D, Flatten, Reshape, Input, Dropout, UpSampling2D, ZeroPadding2D
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
			
		
		stride = 2
		ndf = 1

		
		in_h, in_w, in_d = image_shape
				

		# The convolution
		conv = Sequential(name = 'ConvolutionEncoder')
		conv.add(InputLayer(input_shape = image_shape))
		conv.add(Conv2D(filters = 32, kernel_size = 3, strides = 2, activation='relu', padding = 'same'))
		#conv.add(Dropout(0.1))
		#conv.add(Conv2D(filters = int(ndf / 2), kernel_size = int(ndf / 2), activation='relu', padding = 'same'))
		#conv.add(Dropout(0.01))
		#conv.add(MaxPooling2D(pool_size = 2, strides = 2))
		#conv.add(Conv2D(filters = int(ndf / 1), kernel_size = int(ndf / 1), activation='relu', padding = 'same'))
		#onv.add(Dropout(0.01))
		#conv.add(MaxPooling2D(pool_size = 2, strides = 2))

		aux, en_h, en_w, ndf = conv.output_shape
		print(conv.output_shape)
		conv.add(Flatten())
		print(conv.output_shape)

		# The encoder
		encoder = Sequential(name = 'DenseNetEncoder')
		conv.add(InputLayer(input_shape = (en_h * en_w * ndf,)))
		encoder.add(Dense(encoding_size))
		
		# The decoder
		decoder = Sequential(name = 'DenseNetDecoder')
		decoder.add(InputLayer(input_shape = (encoding_size,)))
		decoder.add(Dense(en_h * en_w * ndf, activation = 'relu'))

		# The deconvolution
		deconv = Sequential(name = 'ConvolutionDecoder')
		deconv.add(InputLayer(input_shape = (en_h * en_w * ndf,)))
		deconv.add(Reshape(target_shape = (en_h, en_w, ndf)))

		#deconv.add(Conv2DTranspose(filters = int(ndf / 1), kernel_size = int(ndf / 1), activation='relu', padding = 'same'))
		deconv.add(UpSampling2D(size= 2, interpolation="nearest"))
		#deconv.add(Conv2DTranspose(filters = 32, kernel_size = 3, activation='relu', padding = 'same'))
		#deconv.add(UpSampling2D(size= 2, interpolation="nearest"))
		#deconv.add(Conv2DTranspose(filters = int(ndf / 4), kernel_size = int(ndf / 4), activation='relu', padding = 'same'))
		#deconv.add(UpSampling2D(size= 2, interpolation="nearest"))
		deconv.add(Conv2DTranspose(filters = 3, kernel_size = 3, strides = 1, padding = 'same'))
		
		#deconv.add(ZeroPadding2D(padding=3))
	

		self.conv = conv
		self.encoder = encoder
		self.decoder = decoder
		self.deconv = deconv

		# Adding all up

		input_obj = Input(shape = image_shape, name = 'Input')
		conv_obj = self.conv(input_obj)
		encoded_obj = self.encoder(conv_obj)
		decoded_obj = self.decoder(encoded_obj)
		reconstr_obj = self.deconv(decoded_obj)

		self.model = Model(input_obj, reconstr_obj)
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

		conv_obj = self.conv.predict(image[None])
		encoded_obj = self.encoder.predict(conv_obj)
		decoded_obj = self.decoder.predict(encoded_obj)
		reconstr_obj = self.deconv.predict(decoded_obj)

		plt.subplot(1,3,1)
		plt.title("Original")
		show_image(image)

		
		plt.subplot(1,3,2)
		plt.title("Encoding")
		
		plt.imshow((encoded_obj.reshape([len(encoded_obj[0]),1]) + 0.5) * 255)
		

		plt.subplot(1,3,3)
		plt.title("Reconstructed")
		show_image(reconstr_obj[0])

		plt.show()

		cv2.imwrite('Result.jpg', cv2.cvtColor((reconstr_obj[0] + 0.5) * 255.0, cv2.COLOR_RGB2BGR))
	
	def get_encoder(self):
		return self.encoder

	def get_decoder(self):
		return self.decoder

	def save_model(self, name):
		self.autoencoder.save(name)