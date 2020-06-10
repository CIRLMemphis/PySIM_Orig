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

import pytorch_ssim
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

#parser = argparse.ArgumentParser(description='Train Super Resolution Models')
#parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
#parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
#                    help='super resolution upscale factor')
#parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


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

def save_pred(epoch,model,test_dataloader):
    model.eval()
    model.train()
    items = next(iter(test_dataloader)) 
    gt = items[1]
    img = items[0]
    pred = model(img)
    loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
    pred = pred.detach().cpu().numpy().astype(np.uint32)
    img = img.detach().cpu().numpy().astype(np.uint32)
    pred = pred[0][0]
    img = img[0][0]
    print('Validation loss : ',loss.item())
    ip = img_proc()
    ip.SaveImg(img,pred)

CROP_SIZE = 128
UPSCALE_FACTOR = 2
NUM_EPOCHS = 10000
batch_size = 64

mf = mat_file()
X_train, X_test, y_train, y_test = mf.get_images()
X_train = np.rollaxis(X_train, 3, 1)
y_train = np.rollaxis(y_train, 3, 1)
X_test = np.rollaxis(X_test, 3, 1)
y_test = np.rollaxis(y_test, 3, 1)
train_data = ImageDataset(X_train,y_train)
test_data = ImageDataset(X_test,y_test)
print(y_train.shape)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=False) # better than for loop

#torch.autograd.set_detect_anomaly(True)
if True:# __name__ == '__main__':
    #opt = parser.parse_args()
    

    
    #train_set = TrainDatasetFromFolder('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    #val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)
    #train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    #val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = UNet(n_channels=15, n_classes=1)
    #print(summary(netG,(15,128,128)))
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    #print(summary(netD,(1,256,256)))

    generator_criterion = GeneratorLoss()

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
    
        netG.train()
        netD.train()
        for data, target in train_bar:
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

   
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
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
        save_pred(epoch,netG,val_loader)