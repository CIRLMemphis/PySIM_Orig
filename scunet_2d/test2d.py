import numpy as np
import torch
from config import *
from tqdm import tqdm
from scipy.io import loadmat, savemat
from unet_model import UNet
import os
import pandas as pd

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
                img = ((inp_img[:, :, 0, i, j]).astype(float))
                if normalize:
                    img_norm = (img - np.min(img))/(np.max(img) - np.min(img))
                inp_set.append(img_norm)
        inp_set = np.array(inp_set).astype(np.float)
        inp_set = np.expand_dims(inp_set, axis=0)
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
        pred = pred.detach().cpu().numpy()[0][0]
        pred = pred.astype(float)
        pred = 255*(pred - np.min(pred))/(np.max(pred) - np.min(pred))
        savemat(file_path, {'crop_g': pred})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet(n_channels=Nthe*Nphi, n_classes=1)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_800_"+str(Nthe)+'_'+str(Nphi)+'.pkl'))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)