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
    min_rang_before = []
    max_rang_before = []
    min_rang_after = []
    max_rang_after = []
    before_dtype = []
    after_dtype =  []
    for i in tqdm(range(1, limit + 1, 1)):
        ni = 6 - len(str(i))
        ni = ''.join(['0'] * ni) + str(i)
        inp_file = inp_fname + ni + '.mat'
        inp_img = loadmat(inp_file)['crop_g']
        inp_set = []
        for i in range(Nthe):
            for j in range(Nphi):
                img = ((inp_img[:, :, 0, i, j]).astype(float))
                min_rang_before.append(np.min(img))
                max_rang_before.append(np.max(img))
                before_dtype.append(img.dtype)
                if normalize:
                    img_norm = ((img / np.max(img)).np.astype(float))
                min_rang_after.append(np.min(img_norm))
                max_rang_after.append(np.max(img_norm))
                after_dtype.append(img_norm.dtype)
                inp_set.append(img_norm)
                col_heads = ['Before Norm - Min', 'Before Norm - Max','Before Norm - DataType', 'After Norm - Min', 'After Norm - Max','After Norm - DataType']
            norm = pd.DataFrame(list(zip(min_rang_before, max_rang_before,before_dtype, min_rang_after, max_rang_after, after_dtype )), columns=col_heads)
            norm.to_csv('Input Data - Dr Van 1002Dataset'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        inp_set = np.array(inp_set).astype('float')
        inp_set = np.expand_dims(inp_set, axis=0)
        inp_set = torch.from_numpy(inp_set).float()
        file_name = os.path.basename(inp_file)
        out_file = os.path.join(out_dir, file_name)
        data.append((inp_set, out_file))
    return data

def save_pred(model, data):
    model.eval()
    pred_min = []
    pred_max = []
    pred_dtype = []
    for image, file_path in data:
        img = image.cuda(cuda)
        pred = model(img)
        pred = pred.detach().cpu().numpy()[0][0]
        pred = (pred).astype(np.float)
        pred_min.append(np.min(pred))
        pred_max.append(np.max(pred))
        pred_dtype.append(pred.dtype)
        pred_heads = ['Pred-Min', 'Pred-Max', 'Pred-Dtype']
        pred_norm = pd.DataFrame(list(zip(pred_min, pred_max,pred_dtype)), columns=pred_heads)
        pred_norm.to_csv('TestData-Predictions_Norm'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')
        savemat(file_path, {'crop_g': pred})

if __name__ == '__main__':
    cuda = torch.device('cuda')
    model = UNet(n_channels=Nthe*Nphi, n_classes=1)
    print("{} Parameters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(model_loc+"Model_Final_550_3_5.pkl"))
    model.eval()
    model.cuda(cuda)
    data = get_images()
    save_pred(model, data)