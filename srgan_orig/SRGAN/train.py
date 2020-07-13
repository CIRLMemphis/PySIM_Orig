import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

from mat_file import mat_file
from torchsummary import summary
from img_proc import img_proc
import numpy as np
from torchvision.models.vgg import vgg16
import sys
from unet_model import UNet
from config import *

def get_learning_rate(epoch):
    learning_rate = 0.001
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

def save_pred(epoch,model,test_dataloader):
    model.eval()
    items = next(iter(test_dataloader)) 
    gt = items[1]
    img = items[0]
    if torch.cuda.is_available():
        img = img.cuda()
    print('###############',gt.shape,img.shape)
    pred = model(img)
    print('**************',gt.shape,img.shape)
    pred = pred.detach().cpu().numpy().astype(np.uint32)
    pred = pred[0][0]

    ip = img_proc()
    gt = gt.detach().cpu().numpy()
    gt = gt[0][0]
    ip.SaveImg(gt,pred)

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

CROP_SIZE = 256
UPSCALE_FACTOR = 2
NUM_EPOCHS = 20000
batch_size = 5
train_loader,val_loader = None,None

if not div_dataset:
    mf = mat_file()
    X_train, X_test, y_train, y_test = mf.get_images()

    X_train = np.rollaxis(X_train, 3, 1)
    y_train = np.rollaxis(y_train, 3, 1)
    X_test = np.rollaxis(X_test, 3, 1)
    y_test = np.rollaxis(y_test, 3, 1)
    X_train = ImageDataset(X_train,y_train)
    y_train = ImageDataset(X_test,y_test)

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
    val_loader = torch.utils.data.DataLoader(y_train, batch_size=batch_size, shuffle=False, pin_memory=False) # better than for loop
    X_train,y_train,X_test,y_test = None,None,None,None

else: 
    train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    #x,y = next(iter(val_loader)),next(iter(train_loader))
    #print(x[0].shape,x[1].shape,x[2].shape,y[0].shape)
#netG = Generator(UPSCALE_FACTOR,in_channels,out_channels) 
netG = UNet(n_channels=in_channels, n_classes=out_channels)
#print(summary(netG,(in_channels,128,128)))
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator(out_channels)
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
#print(summary(netD,(out_channels,256,256)))
generator_criterion = GeneratorLoss()
#print(summary(generator_criterion,(3,256,256)))
if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    lr = get_learning_rate(epoch)
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lr

    netG.train()
    netD.train()
    for data, target in train_bar:
        #print(data.shape)
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()

        print(z.shape)
        fake_img = netG(z)
        fake_out = netD(fake_img).mean()
        nfake_out,nfake_img,nreal_img = fake_out.repeat(1,3,1,1),fake_img.repeat(1,3,1,1),real_img.repeat(1,3,1,1)
        g_loss = generator_criterion(fake_out, nfake_img, nreal_img)
        print(fake_out.shape,nfake_img.shape,nreal_img.shape)
        netG.zero_grad()
        g_loss.backward()
        optimizerG.step()



        real_out = netD(real_img).mean()
        fake_out = netD(fake_img.detach()).mean()
        d_loss = 1 - real_out + fake_out
        netD.zero_grad()
        d_loss.backward(retain_graph=True)
        optimizerD.step()
    

        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    if epoch%10 == 1:
        save_pred(epoch,netG,val_loader)
