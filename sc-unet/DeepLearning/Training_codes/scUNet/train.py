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
#import sys
#path = '/home/star/0_code_lhj/DL-SIM-github/Training_codes/scUNet/'
#sys.path.append(path)

from unet_model import UNet
import sys
from config import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inp_images,out_images):
        self.inp_images = inp_images
        self.out_images = out_images.astype(int)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.inp_images[index].astype(int)).float()
        y = torch.from_numpy(self.out_images[index].astype(int)).float()
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

def save_pred(epoch,model,test_dataloader):
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Evaluating epoch ', epoch)
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
    pred = pred.detach().cpu().numpy().astype(np.uint32)
    pred = pred[0][0]

    ip = img_proc()
    gt = gt.detach().cpu().numpy()
    gt = gt[0][0]
    ip.SaveImg(gt,pred)  







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
X_train = np.rollaxis(X_train, 3, 1)
y_train = np.rollaxis(y_train, 3, 1)
X_test = np.rollaxis(X_test, 3, 1)
y_test = np.rollaxis(y_test, 3, 1) 
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)



train_data = ImageDataset(X_train,y_train)
test_data = ImageDataset(X_test,y_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False) # better than for loop

   

model = UNet(n_channels=in_channels, n_classes=out_channels)
#print(summary(model,(in_channels,128,128)))
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
if have_cuda:
    model.cuda(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

#    loss_all = np.zeros((2000, 4))
c = 0
for epoch in range(2000):
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
