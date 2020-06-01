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

#import sys
#path = '/home/star/0_code_lhj/DL-SIM-github/Training_codes/scUNet/'
#sys.path.append(path)

from unet_model import UNet
import sys





class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inp_images,out_images):
        self.inp_images = inp_images
        self.out_images = out_images.astype(int)
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.inp_images[index]).float()
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
    model.eval()
    model.train()
    items = next(iter(test_dataloader)) 
    gt = items[1]
    img = items[0]
    pred = model(img)
    pred = pred.detach().cpu().numpy().astype(np.uint32)
    img = img.detach().cpu().numpy().astype(np.uint32)
    pred = pred[0][0]
    img = img[0][0]
    ip = img_proc()
    ip.SaveImg(img,pred)




if __name__ == "__main__":
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    have_cuda = True if torch.cuda.is_available() else False
    learning_rate = 0.001
    n_channels = 5
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    mf = mat_file()
    X_train, X_test, y_train, y_test = mf.get_images()
    X_train = np.rollaxis(X_train, 3, 1)
    y_train = np.rollaxis(y_train, 3, 1)
    X_test = np.rollaxis(X_test, 3, 1)
    y_test = np.rollaxis(y_test, 3, 1) 




    train_data = ImageDataset(X_train,y_train)
    test_data = ImageDataset(X_test,y_test)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop
    

    model = UNet(n_channels=n_channels, n_classes=1)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    if have_cuda:
        model.cuda(cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

#    loss_all = np.zeros((2000, 4))
    for epoch in range(2000):
        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))
        for batch_idx, items in enumerate(train_dataloader):
            image = items[0]
            gt = items[1]
            model.train()

            gt = gt.float()
            if have_cuda:
                gt = gt.cuda(cuda)
            
            pred = model(image)

            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ('epoch : ',epoch, 'loss: ',loss.item())
        #if epoch%10 == 5:
            save_pred(epoch,model,test_dataloader)
        torch.save(model.state_dict(), "out/sUNet_microtubule_"+str(epoch+1)+".pkl")
