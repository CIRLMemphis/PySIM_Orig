from xlwt import *
import numpy as np
import os
import math
import torch
from torch.utils.data import DataLoader
from skimage import io, transform
from mat_file import mat_file
from torchsummary import summary
from img_proc import img_proc
from tqdm import tqdm
import sys
import pandas as pd
from skimage.metrics import structural_similarity
from unet_model import UNet
import sys
from config import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inp_images,out_images):
        self.inp_images = inp_images.astype('double')
        self.out_images = out_images.astype('double')
        
    def __getitem__(self, index):
        x = torch.from_numpy(self.inp_images[index].astype('double')).double()
        y = torch.from_numpy(self.out_images[index].astype('double')).double()
        return (x, y)

    def __len__(self):
        return len(self.inp_images)
def get_errors(gt,pr, data_range = None):
    mse = np.mean((gt - pr) ** 2)
    def psnr(gt, pr, mse):
        gt = (gt/np.amax(gt))*255
        pr = (pr/np.amax(pr))*255
        psnr = 20 * math.log10(255/math.sqrt(mse))
        return psnr
    def ssim(gt, pr, data_range = None):
        pr = (pr/pr.max()) * gt.max()
        ssim = structural_similarity(gt, pr, win_size=3, gradient=False, data_range=None, multichannel=True)
        return ssim
    psnr = psnr(gt,pr, mse)
    ssim = ssim(gt,pr)
    metrics = [round(mse,7), round(ssim,7), round(psnr,7)]
    return metrics

def save_pred(epoch,model,test_dataloader):
    cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    metrics = {'loss':0, 'mse1':0, 'ssim1':0, 'psnr1':0,
                'mse2':0, 'ssim2':0, 'psnr2':0,
                'mse3':0, 'ssim3':0, 'psnr3':0}
    
    for batch_idx, items in enumerate(test_dataloader):
        gt = items[1]
        img = items[0]
        if torch.cuda.is_available():
            img = img.cuda(cuda)
        pred = model(img)
        if torch.cuda.is_available():
            pred = pred.cuda(cuda)
            gt = gt.cuda(cuda)
        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        if not (is_3d and convert_to_2d):
            pred = pred[0]
            gt = gt[0]
        pred = pred[0]
        gt = gt[0]
        mse, ssim, psnr = get_errors(gt[0, :, :],pred[0, :, :])
        metrics['mse1'] += mse
        metrics['ssim1'] += ssim
        metrics['psnr1'] += psnr
        mse, ssim, psnr = get_errors(gt[1, :, :],pred[1, :, :])
        metrics['mse2'] += mse
        metrics['ssim2'] += ssim
        metrics['psnr2'] += psnr
        mse, ssim, psnr = get_errors(gt[2, :, :],pred[2, :, :])
        metrics['mse3'] += mse
        metrics['ssim3'] += ssim
        metrics['psnr3'] += psnr
        metrics['loss'] += loss.item()
    ip = img_proc()
    ip.SaveImg(str(epoch)+ '_' + str(metrics['loss']), gt, pred)
    metrics = {k: v/len(test_dataloader) for k,v in metrics.items()}
    return metrics

def save_checkpoint(state,epoch):
    print("=> Saving Checkpoint")
    torch.save( state, model_loc+str(epoch+1)+".pth.tar")

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
have_cuda = True if torch.cuda.is_available() else False
batch_size = 5
X_train,X_test,y_train,y_test = None,None,None,None
mf = mat_file()
inp_images,out_images = mf.get_data()
if use_valid_file: 
    mf.set_valid_dir()
    valid_in,valid_out = mf.get_images()
    X_train,X_test,y_train,y_test = inp_images,valid_in,out_images,valid_out
else:
    X_train,X_test,y_train,y_test = mf.get_test_train(inp_images,out_images)
print('Train_Test_Shape',X_train.shape,y_train.shape,X_test.shape,y_test.shape)
train_data = ImageDataset(X_train,y_train)
test_data = ImageDataset(X_test,y_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop
model = UNet(n_channels=X_train.shape[1], n_classes=y_train.shape[1])
model.double()

start_epoch = 0
if load_model:
    weight = torch.load(model_file)
    model.load_state_dict(weight['model_state_dict'])
    start_epoch = weight['epoch']

print("{} Parameters in Total".format(sum(x.numel() for x in model.parameters())))
print(" See model input shape first:", X_train.shape[1], y_train.shape[1])
if have_cuda:
    model.cuda(cuda)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
train_loss = []
learn_rate =[]
c = 0
val_metrics =  {'loss':[], 'mse1':[], 'ssim1':[], 'psnr1':[],
                'mse2':[], 'ssim2':[],'psnr2':[],
                'mse3':[], 'ssim3':[],'psnr3':[]}
epochs = 3000

for epoch in range(start_epoch, epochs):
    lr = .001 - (epoch/30000)
    if lr < .0000001:
        lr = .00001
    tra_loss = 0
    for batch_idx, items in enumerate(train_dataloader):
        image = items[0]
        gt = items[1]
        model.train()
        gt = gt.double()
        if have_cuda:
            image = image.cuda(cuda)
            gt = gt.cuda(cuda)
        pred = model(image)
        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tra_loss += loss.item()
    tra_loss /= len(train_dataloader)
    metrics = save_pred(epoch,model,test_dataloader)
    train_loss.append(tra_loss)
    for k in metrics.keys():
        val_metrics[k].append(metrics[k])
    learn_rate.append(lr)
    print ('epoch: ',epoch, 'training loss: ', tra_loss, 'LR:', lr)
    epoch_list = range(epochs)
    col_heads = ['Train loss', 'Valid loss', 'MSE-1', 'SSIM-1', 'PSNR-1', 'MSE-2', 'SSIM-2', 'PSNR-2', 'MSE-3', 'SSIM-3', 'PSNR-3', 'LearnRate']
    metrics_assess = pd.DataFrame(list(zip(train_loss, val_metrics['loss'], 
                        val_metrics['mse1'], val_metrics['ssim1'], val_metrics['psnr1'],
                        val_metrics['mse2'], val_metrics['ssim2'], val_metrics['psnr2'],
                        val_metrics['mse3'], val_metrics['ssim3'], val_metrics['psnr3'],
                        learn_rate)), columns=col_heads)
    metrics_assess.to_csv('metrics_3d'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
    if save_model_per_epoch:
        torch.save(model.state_dict(), model_loc+str(epoch+1)+".pkl")        
    if save_model_per_interval:
        if epoch % interval == 0:
            print('=> Saving Checkpoint')
            torch.save(model.state_dict(), model_loc+'Model_Final'+'_'+str(epoch)+'_'+str(Nthe)+'_'+str(Nphi)+".pkl")
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': tra_loss,
                        'learn_rate': lr},    
            model_loc+str(epoch)+".pt")
torch.save(model.state_dict(), model_loc+'Model_Final'+'_'+str(epoch)+'_'+str(Nthe)+'_'+str(Nphi)+".pkl")
torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': tra_loss,
                        'learn_rate': lr},    
            model_loc+str(epoch)+".pt")