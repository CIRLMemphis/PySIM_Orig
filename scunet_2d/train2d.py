from xlwt import *
import numpy as np
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
from mat_file import mat_file
from torchsummary import summary
from img_proc import img_proc
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity
from unet_model import UNet
from config import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.style.use('classic')
plt.figure(figsize=(16, 7))
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, inp_images,out_images):
        self.inp_images = inp_images.astype('float')
        self.out_images = out_images.astype('float')
    def __getitem__(self, index):
        x = torch.from_numpy(self.inp_images[index].astype('float')).float()
        y = torch.from_numpy(self.out_images[index].astype('float')).float()
        return (x, y)
    def __len__(self):
        return len(self.inp_images)

def get_errors(gt,pr, data_range = None):
    mse = np.mean((gt - pr) ** 2)
    def psnr(gt, pr, mse):
        gt = (gt/np.amax(gt))*255
        pr = (pr/np.amax(pr))*255
        psnr = 20 * math.log10(255 / math.sqrt(mse))
        return psnr
    def ssim(gt, pr, data_range = None):
        pr = (pr/pr.max()) * gt.max()
        ssim = structural_similarity(gt, pr, win_size=None, gradient=False, data_range=None, multichannel=True)
        return ssim
    psnr = psnr(gt,pr, mse)
    ssim = ssim(gt,pr)
    metrics = [round(mse,7), round(ssim,7), round(psnr,7)]
    return metrics

def save_pred(epoch,model,test_dataloader):
    cuda = torch.device('cuda')
    model.eval()
    total_loss = 0
    total_mse = 0
    total_ssim = 0
    total_psnr = 0
    for batch_idx, items in enumerate(test_dataloader):
        gt = items[1]
        img = items[0]
        print ("GroundTruth",gt.shape)
        print ("Data",image.shape)
        exit()
        img = img.cuda(cuda)
        pred = model(img)
        pred = pred.cuda(cuda)
        gt = gt.cuda(cuda)
        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        pred = pred.detach().cpu().numpy()
        pred = pred[0][0]
        gt = gt.detach().cpu().numpy()
        gt = np.squeeze(gt[0])
        mse, ssim, psnr = get_errors(gt,pred)
        total_loss += loss.item()
        total_mse += mse
        total_ssim += ssim
        total_psnr += psnr
    totally = 'ok'
    total_loss /= len(test_dataloader)
    total_mse /= len(test_dataloader)
    total_ssim /= len(test_dataloader)
    total_psnr /= len(test_dataloader)
    ip = img_proc()
    ip.SaveImg(str(epoch)+ '_' + str(totally), gt, pred)
    return total_loss, total_mse, total_ssim, total_psnr, 

def save_checkpoint(state,epoch):
    print("=> Saving Checkpoint")
    torch.save(state, model_loc+str(epoch+1)+".pth.tar")        

cuda = torch.device('cuda')
learning_rate = 0.001
batch_size = 12
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
y_train = np.rollaxis(y_train, 3, 1)
y_test = np.rollaxis(y_test, 3, 1) 
print('Test-Train',X_train.shape,y_train.shape,X_test.shape,y_test.shape)
train_data = ImageDataset(X_train,y_train)
test_data = ImageDataset(X_test,y_test)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop  
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=False) # better than for loop
in_channels = X_train.shape[1] 
out_channels = y_train.shape[1]
model = UNet(n_channels=X_train.shape[1], n_classes=y_train.shape[1]) 
print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
model.cuda(cuda)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))
c = 0
valid_mse = []
valid_ssim = []
valid_psnr = []
train_loss = []
valid_loss = []
learn_rate = []
epochs = 1000
for epoch in range(epochs):
    lr = 0.001 - (epoch/30000)
    if lr < 0.000001:
        lr = .00001
    tra_loss = 0
    for batch_idx, items in enumerate(train_dataloader):
        image = items[0]
        gt = items[1]
        model.train()
        image = image.cuda(cuda)
        gt = gt.cuda(cuda)
        pred = model(image)
        loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tra_loss += loss.item()
    tra_loss /= len(train_dataloader)
    val_loss,val_mse, val_ssim, val_psnr = save_pred(epoch,model,test_dataloader) 
    learn_rate.append(lr) 
    train_loss.append(tra_loss)
    valid_loss.append(val_loss)
    valid_mse.append(val_mse)
    valid_ssim.append(val_ssim)
    valid_psnr.append(val_psnr)
    print ('epoch : ',epoch, 'training loss: ', tra_loss, 'Validation loss ', val_loss, 'ssim', val_ssim, 'lr:', lr)
    epoch_list = range(epochs)
    col_heads = ['MSE', 'SSIM', 'PSNR', 'Train loss', 'Validation loss', 'LearnRate']
    metrics_assess = pd.DataFrame(list(zip(valid_mse, valid_ssim, valid_psnr, train_loss, valid_loss, learn_rate)), columns=col_heads)
    metrics_assess.to_csv('metrics_2d'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
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
