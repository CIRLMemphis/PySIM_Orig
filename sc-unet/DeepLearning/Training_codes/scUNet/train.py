#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:56:35 2019

@author: lhjin
"""
from xlwt import *
import numpy as np
import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
from mat_file import mat_file
from torchsummary import summary
from img_proc import img_proc
from tqdm import tqdm
import sys
#path = '/home/star/0_code_lhj/DL-SIM-github/Training_codes/scUNet/'
#sys.path.append(path)

from unet_model import UNet
from unet_3d import UNet3D
import sys
from config import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))

#model = UNet(n_channels=in_channels, n_classes=out_channels)
#print(summary(model,(in_channels,256,256)))
#sys.exit()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inp_images,out_images):
        self.inp_images = inp_images.astype(float)
        self.out_images = out_images.astype(float)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.inp_images[index].astype(float)).float()
        y = torch.from_numpy(self.out_images[index].astype(float)).float()
        return (x, y)

    def __len__(self):
        return len(self.inp_images)

def get_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

def save_input(img):
    img = img[0]
    print(img.shape)
    #print(img.shape)
    n = img.shape[0]
    for i in range(n):
        plt.imshow(img[i])
        plt.savefig(out_dir  + 'inp' + str(i) + '.png')


def get_errors(gt,pr):
    gt = gt.flatten()
    pr = pr.flatten()
    n = len(gt)
    d = sum([(gt[i]-pr[i])**2 for i in range(n)])
    d = d/(n)
    avg_gt = sum(gt)/n
    avg_pr =  sum(pr)/n
    var_gt = sum([(avg_gt-y)**2 for y in gt])
    psnr = 20*math.log10(255/math.sqrt(d))
    nrmse = math.sqrt(d)/var_gt
    r = math.sqrt(d)
    return str(round(r,6)) + '_' + str(round(psnr,4)) + '_' + str(round(nrmse,2))

def save_pred(epoch,model,test_dataloader):
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('Evaluating epoch ', epoch)
    model.eval()
    items = next(iter(test_dataloader)) 
    gt = items[1]
    img = items[0]
    if torch.cuda.is_available():
        img = img.cuda(cuda)
    #print('###############',gt.shape,img.shape)
    pred = model(img)
    if torch.cuda.is_available():
        pred = pred.cuda(cuda)
        gt = gt.cuda(cuda)
    #print('**************',gt.shape,img.shape)
    loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
    print('validation loss : ',loss.item())
    pred = pred.detach().cpu().numpy()
    pred = pred[0][0]
    #for x in pred:
        #print(x)

    ip = img_proc()
    gt = gt.detach().cpu().numpy()
    gt = gt[0]
    if not is_3d:
        gt = gt[0]
    er = get_errors(gt,pred)
    ip.SaveImg(str(epoch) + '_' + er,gt,pred)

    if epoch != 0:
        return
    #save_input(img.detach().cpu().numpy())







cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
have_cuda = True if torch.cuda.is_available() else False
learning_rate = 0.001
# momentum = 0.99
# weight_decay = 0.0001
batch_size = 20
X_train,X_test,y_train,y_test = None,None,None,None
mf = mat_file()
inp_images,out_images = mf.get_data()
if use_valid_file:
    mf.set_valid_dir()
    valid_in,valid_out = mf.get_images()
    X_train,X_test,y_train,y_test = mf.format(inp_images,out_images,valid_in,valid_out)
else:
    X_train,X_test,y_train,y_test = mf.get_test_train(inp_images,out_images)


print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
if is_3d:
    X_test = np.rollaxis(X_test, 4, 2)
    X_train = np.rollaxis(X_train, 4, 2)

y_train = np.rollaxis(y_train, 3, 1)
y_test = np.rollaxis(y_test, 3, 1) 
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)





train_data = ImageDataset(X_train,y_train)
test_data = ImageDataset(X_test,y_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop

   
if is_3d:
    model = UNet3D(n_channels=in_channels, n_classes=out_channels)
else:
    model = UNet(n_channels=in_channels, n_classes=out_channels)
#print(summary(model,(in_channels,256,256)))
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
if have_cuda:
    model.cuda(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

#    loss_all = np.zeros((2000, 4))
c = 0
for epoch in range(5000):
    lr = .001 - (epoch/30000)
    if lr < .00001:
        lr = .00001
    for p in optimizer.param_groups:
        p['lr'] = lr
        print("learning rate = {}".format(p['lr']))

    #print(len(train_dataloader))
    for batch_idx, items in enumerate(train_dataloader):
        #print(epoch)

        image = items[0]
        #print(image.shape)
        gt = items[1]
        model.train()

        gt = gt.float()
        if have_cuda:
            image = image.cuda(cuda)
            gt = gt.cuda(cuda)
        
        pred = model(image)

        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_pred(epoch,model,test_dataloader)
    print ('epoch : ',epoch, 'training loss: ',loss.item())
