from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import math
from skimage.metrics import structural_similarity
from config import *

def get_errors(gt,pr):
    mse = np.mean((gt - pr) ** 2)
    def psnr(gt, pr, mse):
        gt = (gt/np.amax(gt))*255
        pr = (pr/np.amax(pr))*255
        if mse == 0:
            return 100
        else:
            PIXEL_MAX = 255
            psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return psnr
    def ssim(gt, pr):
        ssim = structural_similarity(gt, pr, win_size=None, gradient=False, data_range=None, multichannel=True)
        return ssim
    psnr = psnr(gt,pr, mse)
    ssim = ssim(gt,pr)
    metrics = [round(mse,7), round(ssim,7), round(psnr,7)]
    return metrics

def get_metrics_vol (gt,pred):
    metrics =  {'mse1':0, 'ssim1':0, 'psnr1':0,
                'mse2':0, 'ssim2':0, 'psnr2':0,
                'mse3':0, 'ssim3':0, 'psnr3':0}
    mse, ssim, psnr = get_errors(gt[:, :, 0],pred[:, :, 0])
    metrics['mse1'] += mse
    metrics['ssim1'] += ssim
    metrics['psnr1'] += psnr
    mse, ssim, psnr = get_errors(gt[:, :, 1],pred[:, :, 1])
    metrics['mse2'] += mse
    metrics['ssim2'] += ssim
    metrics['psnr2'] += psnr
    mse, ssim, psnr = get_errors(gt[:, :, 2],pred[:, :, 2])
    metrics['mse3'] += mse
    metrics['ssim3'] += ssim
    metrics['psnr3'] += psnr
    metrics = {k: v/1 for k,v in metrics.items()}
    return metrics

def threed_norm (data):
    pred_norm = []
    for i in range (size_3rd_dim):
        data_i = data[:,:,i]
        pred_i = data[:,:,i]/np.max(data[:,:,i])
        #pred_i[pred_i<0] = 0
        print('RangeBefore:',np.min(data_i),np.max(data_i))
        print('Range:',np.min(pred_i),np.max(pred_i))
        pred_norm.append(pred_i)
    pred_norm_arr = np.array(pred_norm)
    pred_norm_arr = pred_norm_arr.transpose((1, 2, 0))
    pred_norm_arr = (pred_norm_arr).astype(np.double)
    print(pred_norm_arr.shape)
    return pred_norm_arr


######## Main Code ##########
if __name__ == '__main__':
    gt_file = "D:/NNData/Metrics/Three_GT.mat"
    pred_file = "D:/PySIM/scunet_3d/U2Os_Actin_5-7-3x1.mat"
    gt = loadmat(gt_file)['reconOb']
    gt_norm = threed_norm(gt)
    #gt_norm = gt/np.max(gt)
    print('GT-norm',gt_norm.shape,np.min(gt_norm),np.max(gt_norm))
    pred = loadmat(pred_file)['crop_g']
    pred_norm = threed_norm(pred)
    #pred_norm = pred/np.max(pred)
    val_metrics =  {'mse1':[], 'ssim1':[], 'psnr1':[],
                    'mse2':[], 'ssim2':[], 'psnr2':[],
                    'mse3':[], 'ssim3':[], 'psnr3':[]}
    volmetrics = get_errors (gt_norm,pred_norm)
    print('Metrics For 3D Volume',volmetrics)
    metrics = get_metrics_vol(gt_norm,pred_norm)
    for k in metrics.keys():
        val_metrics[k].append(metrics[k])
    col_heads = ['MSE-1', 'SSIM-1', 'PSNR-1', 'MSE-2', 'SSIM-2', 'PSNR-2', 'MSE-3', 'SSIM-3', 'PSNR-3']
    metrics_assess = pd.DataFrame(list(zip( 
                        val_metrics['mse1'], val_metrics['ssim1'], val_metrics['psnr1'],
                        val_metrics['mse2'], val_metrics['ssim2'], val_metrics['psnr2'],
                        val_metrics['mse3'], val_metrics['ssim3'], val_metrics['psnr3'],
                        )), columns=col_heads)
    metrics_assess.to_csv('metrics_3d_data'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')