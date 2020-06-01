
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys
from config import *

class mat_file():
	def __init__(self):
		self.inp_fname = inp_fname
		self.out_fname = out_fname
		self.limit = limit #How many image files we want to train
		self.Nthe = Nthe
		self.Nphi = Nphi


	def get_images(self):
		inp_images,out_images = [],[]
		for i in range(1,self.limit,1):
			ni = 6 - len(str(i))
			ni = ''.join(['0']*ni) + str(i)
			inp_file = self.inp_fname + ni + '.mat'
			out_file = self.out_fname + ni + '.mat'
			inp_img = loadmat(inp_file)['crop_g']
			out_img = loadmat(out_file)['crop_g']
			for j in range(self.Nthe):
				inp_images.append(inp_img[:,:,0,j])
				out_images.append(out_img)

		
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.1)
		print(y_train.shape)
		y_train = np.reshape(y_train, (len(y_train), len(y_train[0]), len(y_train[0]),1))
		y_test = np.reshape(y_test, (len(y_test), len(y_test[0]), len(y_test[0]),1))
		#print(type(X_train))
		return X_train, X_test, y_train, y_test

if __name__ == "__main__":
	mf = mat_file()
	mf.get_images()
