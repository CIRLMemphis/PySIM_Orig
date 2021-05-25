import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat, savemat
from unet_model import UNet
import os
from config import *

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
                for k in range(size_3rd_dim):
                    print('Range of Values Before',np.min(inp_img[:,:,k,i,j]),np.max(inp_img[:,:,k,i,j]))
                    three_stack = inp_img[:,:,k,i,j]/np.max(inp_img[:,:,k,i,j])
                    print('Range of Values After',np.min(three_stack),np.max(three_stack))
                    three_stack.shape
                    imgs.append(three_stack)
                #print(len(imgs))
                # imgs holds 3 pictures per crop
                #if normalize:
                #    imgs = imgs/np.max(imgs)
                inp_set.append(imgs)
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
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)        
        pred = pred.detach().cpu().numpy()[0]
        pred = pred/np.max(pred)
        print('Range of Values',np.min(pred),np.max(pred))
        pred = (pred).astype(np.double)
        pred = pred.transpose((1, 2, 0))
        savemat(file_path, {'crop_g': pred})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet(n_channels=(Nthe*Nphi*3), n_classes=3)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_2500_3_5.pkl"))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)