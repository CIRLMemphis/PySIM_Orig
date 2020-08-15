
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
from tqdm import tqdm

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
		return (inp_images,out_images)	


	def format(self,inp_images,out_images,valid_in,valid_out):
		inp_images = np.array(inp_images)
		valid_in = np.array(valid_in)
		out_images = np.array(out_images)
		out_images = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
		valid_out = np.array(valid_out)
		valid_out = np.reshape(valid_out, (len(valid_out), len(valid_out[0]), len(valid_out[0]),out_channels))
		return inp_images,valid_in,out_images,valid_out

	def get_test_train(self,inp_images,out_images):
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test  = None,None,None,None
		if len(inp_images) == 1:
			y_train = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
			return inp_images,None,y_train,None

		out_images = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.1)
		return X_train, X_test, y_train, y_test

	def get_images(self):
		global mx
		if div_dataset:
			data = self.get_div_images()
			return data
		inp_images,out_images = [],[]
		nmx = 0
		for i in tqdm(range(1,self.limit+1,1)):
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
					x = inp_img[:,:,0,i,j]
					nmx = max(max(y) for y in x)
					if nmx > mx:
						mx = nmx
					inp_set.append(x)
			out_img = loadmat(out_file)['crop_g']
			nmx = max(max(y) for y in out_img)
			if nmx > mx:
				mx = nmx
			out_images.append(out_img)

			inp_images.append(np.transpose(inp_set, (1, 2, 0)))
		print(nmx)
		if nmx > 1:
			inp_images = inp_images/mx
			out_images = out_images/mx
		data = (inp_images,out_images)

		return data

	def get_data(self):
		pfile = pickle_loc + '0.p'
		if os.path.exists(pfile):
			data = self.load()
			return data
		data = self.get_images()
		#self.store(data)
		return data

	def load(self):
		files = os.listdir(pickle_loc)
		x,y = [],[]
		for f in files:
			f = pickle_loc + f
			d = pickle.load(open(f, "rb" ))
			x += d[0]
			y += d[1]

		x,y = np.array(x),np.array(y)
		print(x.shape,y.shape)
		return x,y



	def store(self,data):
		n = len(data[0])
		it = n//pickle_n
		for i in range(pickle_n):
			x = data[0][i*it:(i+1)*it]
			y = data[1][i*it:(i+1)*it]
			pfile = pickle_loc + str(i) + '.p'
			pickle.dump((x,y),open(pfile,"wb"))
			x,y = np.array(x),np.array(y)




if __name__ == "__main__":
	mf = mat_file()
	mf.get_images()
