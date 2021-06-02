from scipy.io import loadmat
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys
from config import *
import os
import pickle
import os
from img_proc import img_proc
import matplotlib.pyplot as plt
from tqdm import tqdm

class mat_file():
	def __init__(self):
		self.inp_fname = inp_fname
		self.out_fname = out_fname
		self.limit = limit 
		self.Nthe = Nthe
		self.Nphi = Nphi

	def set_valid_dir(self):
		global valid_in,valid_out,valid_limit
		self.inp_fname = valid_in
		self.out_fname = valid_out
		self.limit = valid_limit
	
	def format(self,inp_images,out_images,valid_in,valid_out):
		inp_images = np.array(inp_images)
		valid_in = np.array(valid_in)
		out_images = np.array(out_images)
		valid_out = np.array(valid_out)
		if not is_3d:
			out_images = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
			valid_out = np.reshape(valid_out, (len(valid_out), len(valid_out[0]), len(valid_out[0]),out_channels))
		return inp_images,valid_in,out_images,valid_out

	def get_test_train(self,inp_images,out_images):
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		X_train, X_test, y_train, y_test  = None,None,None,None
		if len(inp_images) == 1:
			y_train = np.reshape(out_images, (len(out_images), len(out_images[0]), len(out_images[0]),out_channels))
			return inp_images,None,y_train,None
		X_train, X_test, y_train, y_test = train_test_split(inp_images, out_images, test_size=0.10)
		return X_train, X_test, y_train, y_test

	def get_images(self):
		inp_images,out_images = [],[]
		for i in tqdm(range(1,self.limit+1,1)):
			ni = 6 - len(str(i))
			ni = ''.join(['0']*ni) + str(i)
			inp_file = self.inp_fname + ni + '.mat'
			out_file = self.out_fname + ni + '.mat'
			inp_img  = loadmat(inp_file)['crop_g']
			inp_set  = []
			for i in range(self.Nthe):
				for j in range(self.Nphi):
					imgs = []
					if is_3d:
						for k in range(size_3rd_dim):
							three_stack = inp_img[:,:,k,i,j]/np.max(inp_img[:,:,k,i,j]) 
							imgs.append(three_stack)
					else:
						imgs = inp_img[:,:,0,i,j]/np.max(inp_img[:,:,0,i,j])
					inp_set.append(imgs)
			out_img = loadmat(out_file)['crop_g']
			s = out_img.shape
			if len(s) == 3:
				out_img = out_img.transpose((2, 0, 1))
			imgs = []
			if is_3d:
				for k in range(size_3rd_dim):
					imgs.append(out_img[k]/np.max(out_img[k]))
			else:
				imgs = out_img/np.max(out_img)			
			out_images.append([imgs])
			inp_images.append(inp_set)
		inp_images,out_images = np.array(inp_images),np.array(out_images)
		if convert_to_2d:
			inp_images,out_images = self.get_2d_converted_data(inp_images,out_images)
		data = (inp_images,out_images)
		return data
		
	def save(self,out_images):
		print("Out-Images Shape", out_images.shape)
		for i in range(3):
			ofile2 = str(i) + '.png'
			plt.figure(figsize=(8, 3.5))
			plt.imshow(out_images[i])
			plt.savefig(ofile2)
	def get_data(self):
		data = self.get_images()
		self.store(data)
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
		print("Data_Load",x.shape,y.shape)
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

	def get_2d_converted_data(self,inp_images,out_images):
		si,so = inp_images.shape,out_images.shape
		print("Converted 2D Shape",si,so)
		inp_images = np.reshape(inp_images,(si[0],si[1]*si[2],si[3],si[4]))
		out_images = np.reshape(out_images,(so[0],so[1]*so[2],so[3],so[4]))
		return (inp_images,out_images)

	def get_3d_converted_data(self,imgs,ch,d):
		si = imgs.shape
		imgs = np.reshape(imgs,(si[0],int(si[1]/d),int(si[1]/ch),si[2],si[3]))
		return imgs


if __name__ == "__main__":
	mf = mat_file()
	mf.get_images()
