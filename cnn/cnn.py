import tensorflow as tf
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from mat_files_proc import mat_files_proc
import numpy as np
from keras.utils import to_categorical
from img_proc import img_proc

class PredictionCallback(tf.keras.callbacks.Callback):    
  def on_epoch_end(self, epoch, logs={}):
  	if epoch%10 != 5:
  		return
  	y_pred = self.model.predict(self.validation_data[0])
  	act_img = self.validation_data[0][0]
  	pred_img = y_pred[0]

  	#return
  	ip = img_proc()
  	ip.SaveImg(act_img,pred_img)


class CNN:
	def __init__(self):
		self.height = 128
		self.width = 128
		self.channels = 5
		self.build_model()

	def build_model(self):
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(self.height,self.width,self.channels),padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D((2, 2),padding='same'))

		model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

		model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))                  
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

		model.add(Conv2DTranspose(64, kernel_size = (3,3), padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(UpSampling2D())

		model.add(Conv2DTranspose(32, kernel_size = (3,3), padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(UpSampling2D())

		model.add(Conv2DTranspose(16, kernel_size = (3,3), padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(UpSampling2D())

		model.add(Conv2DTranspose(1, kernel_size = (3,3), padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(UpSampling2D())
		
		self.model = model
		print(model.summary())

	def train(self):
		mfp = mat_files_proc()
		X_train, X_test, y_train, y_test = mfp.get_images()
		model = self.model
		model.compile(loss='mse',optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
		model.fit(X_train,y_train,epochs=1000,callbacks=[PredictionCallback()],validation_data=(X_test, y_test))
		p = model.predict(inp_images[0])
		print(p)



	def test(self):
		pass

cnn = CNN()
cnn.train()
