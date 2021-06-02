from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import math
from skimage.metrics import structural_similarity
from config import *

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

######## Main Code ##########
gt_file = "D:/NNData/Metrics/Two_GT.mat"
pred_file = "D:/PySIM/scunet_2d_FairSIM/2D_U2Os_Actin_3x1.mat"
gt = loadmat(gt_file)['reconOb']
print("min,load,gt",np.max(gt))
gt = gt / np.max(gt)
pred = loadmat(pred_file)['crop_g']
pred = pred / np.max(pred)
val_metrics =  {'mse1':[], 'ssim1':[], 'psnr1':[],
                'mse2':[], 'ssim2':[], 'psnr2':[],
                'mse3':[], 'ssim3':[], 'psnr3':[]}
volmetrics = get_errors (gt,pred)
print('Metrics For 3D Volume',volmetrics)
metrics = get_metrics_vol(gt,pred)
for k in metrics.keys():
    val_metrics[k].append(metrics[k])
col_heads = ['MSE-1', 'SSIM-1', 'PSNR-1', 'MSE-2', 'SSIM-2', 'PSNR-2', 'MSE-3', 'SSIM-3', 'PSNR-3']
metrics_assess = pd.DataFrame(list(zip( 
                    val_metrics['mse1'], val_metrics['ssim1'], val_metrics['psnr1'],
                    val_metrics['mse2'], val_metrics['ssim2'], val_metrics['psnr2'],
                    val_metrics['mse3'], val_metrics['ssim3'], val_metrics['psnr3'],
                    )), columns=col_heads)
metrics_assess.to_csv('metrics_Assess_2d'+'_'+str(Nthe)+'_'+str(Nphi)+'.csv')