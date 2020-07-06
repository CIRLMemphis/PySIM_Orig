
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys
from config import *
import os
import cv2

class mat_file():
	def __init__(self):
		self.inp_fname = inp_fname
		self.out_fname = out_fname
		self.limit = limit #How many image files we want to train
		self.Nthe = Nthe
		self.Nphi = Nphi

	def get_div_images(self):
		files = os.listdir(div_lr)
		inp_images,out_images = [],[]
		for f in files:
			fname = div_lr + f
			im = cv2.imread(fname)
			inp_images.append(im)

		for f in files:
			fname = div_hr + f
			im = cv2.imread(fname)
			out_images.append(im)

		inp_images,out_images = inp_images[:limit],out_images[:limit]
		return self.get_test_train(inp_images,out_images)	


	def get_test_train(self,inp_images,out_images):
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.1)
		y_train = np.reshape(y_train, (len(y_train), len(y_train[0]), len(y_train[0]),out_channels))
		y_test = np.reshape(y_test, (len(y_test), len(y_test[0]), len(y_test[0]),out_channels))
		return X_train, X_test, y_train, y_test

	def get_images(self):
		if div_dataset:
			return self.get_div_images()
		inp_images,out_images = [],[]
		for i in range(1,self.limit,1):
			ni = 6 - len(str(i))
			ni = ''.join(['0']*ni) + str(i)
			inp_file = self.inp_fname + ni + '.mat'
			out_file = self.out_fname + ni + '.mat'
			inp_img  = loadmat(inp_file)['crop_g']
			inp_set  = []
			for i in range(self.Nthe):
				for j in range(self.Nphi):
					inp_set.append(inp_img[:,:,0,i,j])
			out_img = loadmat(out_file)['crop_g']
			out_images.append(out_img)
			inp_images.append(np.transpose(inp_set, (1, 2, 0)))

		return self.get_test_train(inp_images,out_images)


if __name__ == "__main__":
	mf = mat_file()
	mf.get_images()
