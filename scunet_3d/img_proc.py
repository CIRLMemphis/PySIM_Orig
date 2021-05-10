import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
import os.path
import scipy

plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))

class img_proc:
	def __init__(self):
		self.out_dir = out_dir

	def SaveImg(self,epoch,act_img,pred_img):
		if is_3d: #(,convert_to_2d):
			self.Save3DImg(epoch,act_img,pred_img)
		return
		#if (is_3d and not convert_to_2d): this initially didn't change things when convert to 2d is false
		#	return
		
		ofile2 = self.out_dir + 'pred' + str(epoch) + '.png'
		plt.figure(figsize=(8, 3.5))

		plt.imshow(pred_img)
		plt.savefig(ofile2)

		ofile1 = self.out_dir + 'gt.png'
		if os.path.isfile(ofile1):
			return
		plt.imshow(act_img)
		plt.savefig(ofile1)


	def Save3DImg(self,epoch,act_img,pred_img):
		#print(act_img.shape,pred_img.shape)
		for i in range(size_3rd_dim):
			ofile2 = self.out_dir + 'pred' + str(epoch) + '_' + str(i) + '.png'
			plt.figure(figsize=(8, 3.5))

			plt.imshow(pred_img[i])
			plt.savefig(ofile2)

			ofile1 = self.out_dir + '_' + str(i) + 'gt.png'
			plt.imshow(act_img[i])
			plt.savefig(ofile1)
		if save_mat:
			ofile3 = self.out_dir + 'pred' + str(epoch) + '_' + str(i) + '.mat'
			scipy.io.savemat(ofile3, {'crop_g': pred_img[i]})

			ofile4 = self.out_dir + '_' + str(i) + 'gt.mat'
			scipy.io.savemat(ofile4, {'crop_g': act_img[i]})

