
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys
from config import *
import os
import cv2
import pickle
import os
from img_proc import img_proc
import matplotlib

class mat_file():
	def __init__(self):
		self.inp_fname = inp_fname
		self.out_fname = out_fname
		self.limit = limit #How many image files we want to train
		self.Nthe = Nthe
		self.Nphi = Nphi


	def set_valid_dir(self):
		global valid_in,valid_out,valid_limit
		self.inp_fname = valid_in
		self.out_fname = valid_out
		self.limit = valid_limit

	def get_div_images(self):
		files = os.listdir(div_lr)
		inp_images,out_images = [],[]
		l = None
		for f in files:
			fname = div_lr + f
			im = cv2.imread(fname)
			if not l:
				l = len(im)
			nim = [[[im[i][j][0]*.001172 + im[i][j][1]*.002302 + im[i][j][2]]*.000447 for j in range(l)] for i in range(l)]
			inp_images.append(nim)
		l = None
		for f in files:
			fname = div_hr + f
			im = cv2.imread(fname)
			if not l:
				l = len(im)
			nim = [[[im[i][j][0]*.001172 + im[i][j][1]*.002302 + im[i][j][2]]*.000447 for j in range(l)] for i in range(l)]
			out_images.append(nim)

		print('preprocessed images')
		inp_images,out_images = inp_images[:limit],out_images[:limit]
		return self.get_test_train(inp_images,out_images)	


	def get_test_train(self,inp_images,out_images):
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test  = None,None,None,None
		print(len(inp_images))
		if len(inp_images) == 1:
			y_train = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
			return inp_images,None,y_train,None
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.01)
		y_train = np.reshape(y_train, (len(y_train), len(y_train[0]), len(y_train[0]),out_channels))
		y_test = np.reshape(y_test, (len(y_test), len(y_test[0]), len(y_test[0]),out_channels))
		#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
		return X_train, X_test, y_train, y_test

	def get_images(self):
		pfile = 'data.p'
		if os.path.exists(pfile):
			return pickle.load(open(pfile, "rb" ))
		if div_dataset:
			data = self.get_div_images()
			pickle.dump(data,open(pfile,"wb"))
			return data
		inp_images,out_images = [],[]
		for i in range(1,self.limit+1,1):
			if i == 436:
				continue
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

		data = self.get_test_train(inp_images,out_images)
		#pickle.dump(data,open(pfile,"wb"))
		return data


if __name__ == "__main__":
	mf = mat_file()
	mf.get_images()
