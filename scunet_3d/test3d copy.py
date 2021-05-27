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
                    three_stack = inp_img[:,:,k,i,j]/np.max(inp_img[:,:,k,i,j])
                    min_rang_after.append(np.min(three_stack))
                    max_rang_after.append(np.max(three_stack))
                    imgs.append(three_stack)
                inp_set.append(imgs)
            col_heads = ['Before-Min', 'Before-Max', 'After-Min', 'After-Max']
            norm = pd.DataFrame(list(zip(min_rang_before, max_rang_before,min_rang_after, max_rang_after, )), columns=col_heads)
            norm.to_csv('TestLog_mat'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
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
    pred_3_min =[]
    pred_3_max =[]
    pred_norm = []
    pred_norm_min = []
    pred_norm_max = []
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)        
        pred = pred.detach().cpu().numpy()[0]
        pred = (pred).astype(np.double)
        pred = pred.transpose((1, 2, 0))
        pred_min.append(np.min(pred))
        pred_max.append(np.max(pred))
        pred_three = pred/np.max(pred)
        pred_3_min.append(np.min(pred_three))
        pred_3_max.append(np.max(pred_three))
        for i in range (size_3rd_dim):
            pred_i = pred[:,:,i]/np.max(pred[:,:,i])
            pred_norm_min.append(np.min(pred_i))
            pred_norm_max.append(np.max(pred_i))
            pred_norm.append(pred_i)
        pred_norm_arr = np.array(pred_norm)
        pred_norm_arr = pred_norm_arr.transpose((1, 2, 0))
        pred_heads = ['Pred-Min', 'Pred-Max','Pred-Img-Norm-Min', 'Pred-Img-Norm-Max','Pred-3Img-Norm-Min', 'Pred-3Img-Norm-Max']
        pred_data = pd.DataFrame(list(zip(pred_min, pred_max,pred_norm_min, pred_norm_max,pred_3_min,pred_3_max )), columns=pred_heads)
        pred_data.to_csv('TestLog'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        savemat(file_path, {'crop_g': pred_norm_arr})

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