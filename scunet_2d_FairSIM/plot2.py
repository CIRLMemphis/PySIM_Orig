import numpy as np
from glob import glob 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
res    = loadmat('drive/My Drive/Colab Notebooks/U2OSActin000007.mat')
crop_g = res['crop_g']

res    = loadmat('drive/My Drive/Colab Notebooks/outU2OSActin000007.mat')
out_ob = res['crop_g']

def read_mat_files(mat_folderpath):
    mat_filepaths = glob(mat_folderpath + '*')
    mat_filepaths = sorted(mat_filepaths)
    print("Total {} .mat files found".format(len(mat_filepaths)))
    img_crops_list = []
    for filepath in mat_filepaths :
        data = loadmat(filepath)
        arr = data['crop_g']
        img_crops_list.append(arr)
    img_crops_arr =  np.array(img_crops_list)
    return img_crops_arr

mat_folderpath = "D:/NNData/NNData_0626/test_result/" # update the mat folder location as per your system. use forward slash / on mac/linux systems
img_crops_arr  =  read_mat_files(mat_folderpath)

# showing 15 images of one sample input file
plt.style.use('classic')
plt.figure(figsize=(16, 7))
cnt  = 1
for i in range(len(img_crops_arr)):
    plt.subplot(3,5,cnt)
    plt.imshow(crop_g, cmap='jet')
    cnt = cnt + 1
plt.show()