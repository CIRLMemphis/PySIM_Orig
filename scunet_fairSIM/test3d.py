import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat, savemat
from unet_model import UNet
import os
from config import *
import pandas as pd

def get_2d_converted_data(inp_images):
    si = inp_images.shape
    inp_images = np.reshape(inp_images,(si[0],si[1]*si[2],si[3],si[4]))
    return (inp_images)

def get_images():
    data = []
    min_rang_before = []
    max_rang_before = []
    min_rang_after = []
    max_rang_after = []
    for i in tqdm(range(1, limit + 1, 1)):
        ni = 6 - len(str(i))
        ni = ''.join(['0'] * ni) + str(i)
        inp_file = inp_fname + ni + '.mat'
        inp_img = loadmat(inp_file)['crop_g']
        inp_set = []
        for i in range(Nthe):
            for j in range(Nphi):
                imgs = []
                for k in range(size_3rd_dim):
                    init = inp_img[:,:,k,i,j]
                    min_rang_before.append(np.min(init))
                    max_rang_before.append(np.max(init))
                    #print('Range of Values Before',np.min(inp_img[:,:,k,i,j]),np.max(inp_img[:,:,k,i,j]))
                    three_stack = inp_img[:,:,k,i,j]/np.max(inp_img[:,:,k,i,j])
                    min_rang_after.append(np.min(three_stack))
                    max_rang_after.append(np.max(three_stack))
                    #print('Range of Values After',np.min(three_stack),np.max(three_stack))
                    #three_stack.shape
                    imgs.append(three_stack)
                #print(len(imgs))
                # imgs holds 3 pictures per crop
                #if normalize:
                #    imgs = imgs/np.max(imgs)
                inp_set.append(imgs)
            col_heads = ['Before-Min', 'Before-Max', 'After-Min', 'After-Max']
            norm = pd.DataFrame(list(zip(min_rang_before, max_rang_before,min_rang_after, max_rang_after, )), columns=col_heads)
            norm.to_csv('TrainData-DataLoad_Norm'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        inp_set = np.array(inp_set)
        inp_set = np.expand_dims(inp_set, axis=0)
        if convert_to_2d:
            inp_set = get_2d_converted_data(inp_set)
        inp_set = torch.from_numpy(inp_set).float()
        file_name = os.path.basename(inp_file)
        out_file = os.path.join(out_dir, file_name)
        data.append((inp_set, out_file))
    return data

def save_pred(model, data):
    model.eval()
    pred_min = []
    pred_max = []
    pred_min_norm = []
    pred_max_norm = []
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)        
        pred = pred.detach().cpu().numpy()[0]
        pred = (pred).astype(np.double)
        pred = pred.transpose((1, 2, 0))
        pred_norm = pred/np.max(pred)
        pred_min.append(np.min(pred))
        pred_max.append(np.max(pred))
        pred_min_norm.append(np.min(pred_norm))
        pred_max_norm.append(np.max(pred_norm))
        pred_heads = ['Pred-Min', 'Pred-Max','PredNorm-Min', 'PredNorm-Max']
        pred_norm = pd.DataFrame(list(zip(pred_min, pred_max,pred_min_norm, pred_max_norm)), columns=pred_heads)
        pred_norm.to_csv('TrainData-Predictions_Norm_Check'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        savemat(file_path, {'crop_g': pred})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet(n_channels=(Nthe*Nphi*3), n_classes=3)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_550_3_5.pkl"))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)