import tensorflow as tf
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten,merge,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,Conv2DTranspose,UpSampling2D,Add
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from mat_files_proc import mat_files_proc
import numpy as np
from keras.utils import to_categorical
from img_proc import img_proc
import tensorlayer as tl
import keras.backend as kb


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




class UNet:
	def __init__(self):
		self.height = 128
		self.width = 128
		self.channels = 5
		self.kernel_size = (3, 3)
		self.act1 = 'linear'
		self.pad = 'same'
		self.pool_size = (2,2)
		self.alpha = .1
		self.shape = (self.height,self.width,self.channels)
		self.build_model()

	def double_conv(self,inp,n_chan):
		out1 = Conv2D(n_chan, self.kernel_size, activation=self.act1,padding=self.pad)(inp)
		out = BatchNormalization()(out1)
		out = LeakyReLU(alpha=self.alpha)(out)
		out = Conv2D(n_chan, self.kernel_size, activation=self.act1,padding=self.pad)(out)
		out = BatchNormalization()(out)
		out = LeakyReLU(alpha=self.alpha)(out)
		out = Add()([out1,out])
		return out

	def down(self,inp,n_chan):
		out = MaxPooling2D(pool_size=self.pool_size,padding=self.pad)(inp)
		out = self.double_conv(out,n_chan)
		return out

	def up(self,inp,n_chan):
		out = Conv2DTranspose(n_chan,kernel_size=self.kernel_size,padding=self.pad)(inp)
		out = UpSampling2D()(out)
		out = self.double_conv(out,n_chan)
		return out		


	def build_model(self):
		inp = Input(self.shape)
		out = Conv2D(64, kernel_size=(3, 3),activation='linear',input_shape=(self.height,self.width,self.channels),padding='same')(inp)
		out = self.down(out,128)
		out = self.down(out,256)
		out = self.down(out,512)
		out = self.down(out,1024)
		out = self.up(out,512)
		out = self.up(out,256)
		out = self.up(out,128)
		out = self.up(out,64)
		out = self.up(out,1)

		model = Model(inputs=[inp],outputs=out)
		print(model.summary())
		self.model = model

	def custom_loss_function(self, y_actual, y_predicted):
		custom_loss_value = kb.mean(kb.sum(5*kb.square(y_actual - y_predicted) + (y_actual - y_predicted) ))
		return custom_loss_value

	def train(self):
		mfp = mat_files_proc()
		X_train, X_test, y_train, y_test = mfp.get_images()
		model = self.model
		model.compile(loss=self.custom_loss_function,optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
		model.fit(X_train,y_train,epochs=1000,callbacks=[PredictionCallback()],validation_data=(X_test, y_test))
		p = model.predict(inp_images[0])
		print(p)



	def test(self):
		pass

if __name__ == "__main__":
	cnn = UNet()
	cnn.train()
