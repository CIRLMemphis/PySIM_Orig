import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat, savemat
from unet_3d import UNet3D
#from unet_model import UNet
import os
from config import *
import pandas as pd

def get_2d_converted_data(inp_images):
    si = inp_images.shape
    inp_images = np.reshape(inp_images,(si[0],si[1]*si[2],si[3],si[4]))
    return (inp_images)

def get_images():
    data = []
    for i in tqdm(range(1, limit + 1, 1)):
        ni = 6 - len(str(i))
        ni = ''.join(['0'] * ni) + str(i)
        inp_file = inp_fname + ni + '.mat'
        inp_img = loadmat(inp_file)['crop_g']
        inp_set = []
        for i in range(Nthe):
            for j in range(Nphi):
                imgs = []
                if is_3d:
                    for k in range(size_3rd_dim):
                        imgs.append(inp_img[:,:,k,i,j])
                imgs = (imgs - np.min(imgs))/(np.max(imgs) - np.min(imgs))
                inp_set.append(imgs)
        inp_set = np.array(inp_set)
        #print(inp_set.shape)
        #inp_set = inp_set.transpose((2, 3, 0, 1))
        #print(inp_set.shape)
        #exit()
        inp_set = np.expand_dims(inp_set, axis=0)
        #if convert_to_2d:
        #    inp_set = get_2d_converted_data(inp_set)
        inp_set = torch.from_numpy(inp_set.astype('float')).float()
        #print('after',inp_set.shape)
        file_name = os.path.basename(inp_file)
        out_file = os.path.join(out_dir, file_name)
        data.append((inp_set, out_file))
    return data

def save_pred(model, data):
    model.eval()
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)        
        pred = pred.detach().cpu().numpy().astype(np.float)[0]
        
        pred = pred.transpose((2, 3, 0, 1))
        pred = pred.astype(float)
        pred = 255*(pred - np.min(pred))/(np.max(pred) - np.min(pred))
        savemat(file_path, {'crop_g': pred})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet3D(n_channels=(15), n_classes=3)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_1400_"+str(Nthe)+'_'+str(Nphi)+'.pkl'))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)