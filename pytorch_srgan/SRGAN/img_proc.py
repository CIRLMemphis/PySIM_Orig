import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
import os.path

plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))

class img_proc:
	def __init__(self):
		self.out_dir = out_dir

	def SaveImg(self,act_img,pred_img):
		#act_img  = np.reshape(act_img, (len(act_img), len(act_img)))
		#pred_img = np.reshape(pred_img, (len(pred_img), len(pred_img)))
		l = os.listdir(self.out_dir)
		c = len(l)

		ofile2 = self.out_dir + 'pred_img' + str(c) + '.png'
		plt.figure(figsize=(8, 3.5))

		plt.imshow(pred_img)
		plt.savefig(ofile2)

		ofile1 = self.out_dir + 'act_img.png'
		if os.path.isfile(ofile1):
			return
		plt.imshow(act_img)
		plt.savefig(ofile1)