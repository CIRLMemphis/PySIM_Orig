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
                    imgs.append(init)
                if normalize:
                    imgs = imgs/(np.max(imgs))
                inp_set.append(imgs)
            col_heads = ['Before-Min', 'Before-Max', 'After-Min', 'After-Max']
            norm = pd.DataFrame(list(zip(min_rang_before, max_rang_before,min_rang_after, max_rang_after, )), columns=col_heads)
            norm.to_csv('TestLog_mat'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        inp_set = np.array(inp_set)
        inp_set = np.expand_dims(inp_set, axis=0)
        if convert_to_2d:
            inp_set = get_2d_converted_data(inp_set)
        inp_set = torch.from_numpy(inp_set.astype('double')).double()
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
    pred_norm_min = []
    pred_norm_max = []
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)        
        pred = pred.detach().cpu().numpy().astype(np.double)[0]
        #print(pred.shape,pred.dtype)
        pred = pred.transpose((1, 2, 0))
        pred_min.append(np.min(pred))
        pred_max.append(np.max(pred))
        pred_three = pred/np.max(pred)
        pred_3_min.append(np.min(pred_three))
        pred_3_max.append(np.max(pred_three))
        pred_norm = []
        for i in range (size_3rd_dim):
            pred_i = pred[:,:,i]/np.max(pred[:,:,i])
            pred_norm_min.append(np.min(pred_i))
            pred_norm_max.append(np.max(pred_i))
            pred_norm.append(pred_i)
           
        #print(len(pred_norm))   
        pred_norm_arr = np.array(pred_norm)
        pred_norm_arr = pred_norm_arr.transpose((1, 2, 0))
        pred_norm_arr = (255*pred_norm_arr).astype(np.double)
        #print(pred_norm_arr.shape)
        
        pred_heads_1 = ['Pred-Min', 'Pred-Max','Pred-3Img-Norm-Min', 'Pred-3Img-Norm-Max']
        pred_heads_2 = ['Pred-Img-Norm-Min', 'Pred-Img-Norm-Max']
        pred_data1 = pd.DataFrame(list(zip(pred_min, pred_max,pred_3_min,pred_3_max )), columns=pred_heads_1)
        pred_data2 = pd.DataFrame(list(zip(pred_norm_min, pred_norm_max, )), columns=pred_heads_2)
        pred_data1.to_csv('PredTotalNorm'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        pred_data2.to_csv('PredPerPicNorm'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        savemat(file_path, {'crop_g': pred_norm_arr})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet(n_channels=(Nthe*Nphi*3), n_classes=3)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_550_3_1.pkl"))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)
