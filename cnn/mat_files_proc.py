
from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split

class mat_files_proc:
	def __init__(self):
		self.inp_fname = '../../in/inFairSIM'
		self.out_fname = '../../out/outFairSIM'
		self.limit = 1002 #How many image files we want to train
		self.Nthe = 3
		self.Nphi = 5


	def get_images(self):
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

		
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.1)
		y_train = np.reshape(y_train, (len(y_train), len(y_train[0]), len(y_train[0]),1))
		y_test = np.reshape(y_test, (len(y_test), len(y_test[0]), len(y_test[0]),1))
		return X_train, X_test, y_train, y_test



