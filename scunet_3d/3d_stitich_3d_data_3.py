# importing required libraries
import numpy as np
from glob import glob 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 

def read_mat_files(mat_folderpath):
    mat_filepaths = glob(mat_folderpath + '*')
    mat_filepaths = sorted(mat_filepaths)
    print("Total {} .mat files found".format(len(mat_filepaths)))
    print("Sample filepaths:{}".format(mat_filepaths[:5]))
    img_crops_list = []
    for filepath in mat_filepaths :
        data = loadmat(filepath)
        arr = data['crop_g']
        img_crops_list.append(arr)
    #print(arr.dtype, arr.shape) # isaw (256, 256, 3)
    img_crops_arr =  np.array(img_crops_list)
    return img_crops_arr

def get_summed_img(img_crops_arr, step_h, step_w):
    #img_crops_arr[img_crops_arr<0] = 0
    num_crops_h, num_crops_w, crop_h, crop_w, crop_c = img_crops_arr.shape
    img_summed_h = crop_h + (num_crops_h-1)*step_h
    img_summed_w = crop_w + (num_crops_w-1)*step_w
    img_summed = np.zeros( (img_summed_h , img_summed_w, crop_c) , dtype=np.float32)
    mask = np.zeros( (img_summed_h , img_summed_w, crop_c) , dtype=np.float32)
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            img_summed[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, : ] += img_crops_arr[idx_h, idx_w, :].astype(np.float32)
            mask[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w , :] += 1
    return img_summed, mask

def matsave_stitched_img(stitched_img, key_name='crop_g', out_filepath='U2Os_Actin_5-7-3x1-3D.mat'):
    out_dict = {key_name:stitched_img}
    savemat(out_filepath, out_dict)

# MAIN CODE 
mat_folderpath = "D:/NNData/3D/FairSIM3D_082420/test_result/" 
img_crops_arr =  read_mat_files(mat_folderpath)
print("Shape of img_crops_arr is {}. Dtype is {}".format(img_crops_arr.shape, img_crops_arr.dtype))
step_h = 64
step_w = 64
stitched_img_h = 1024
stitched_img_w = 1024

total_num_crops = img_crops_arr.shape[0]
num_crops_total, crop_h, crop_w, crop_c  = img_crops_arr.shape
print(num_crops_total)
num_crops_h = (stitched_img_h - crop_h)//step_h + 1
num_crops_w = (stitched_img_w - crop_w)//step_w + 1
print("Number of crops in height direction: {} , number of crops in width direction: {}".format(num_crops_h, num_crops_w))
# error checking like Dr Van's
#assert num_crops_total == num_crops_h * num_crops_w 
#assert stitched_img_h  == crop_h + (num_crops_h-1)*step_h
#assert stitched_img_w  == crop_w + (num_crops_w-1)*step_w
img_crops_arr = img_crops_arr.reshape(num_crops_h, num_crops_w, crop_h,crop_w, crop_c)
img_summed, mask = get_summed_img(img_crops_arr, step_h, step_w) 

print(img_summed.min())
stitched_img = img_summed / mask
#stitched_img = np.round(255 * (stitched_img/stitched_img.max())) # rounding to nearest integer value 
stitched_img = np.round(stitched_img)
stitched_img = stitched_img.astype('uint16') # converting to dtype as uint16 - same as the dtype of cropped images 
#stitched_img = np.squeeze(stitched_img)
#stitched_img = 65535 * (stitched_img - stitched_img.min()) / (stitched_img.max() - stitched_img.min())
#stitched_img = stitched_img.astype(np.uint8).astype(np.uint16) # converting to dtype as uint16 - same as the dtype of cropped images 
#stitched_img = np.squeeze(stitched_img)
print(stitched_img.shape)
matsave_stitched_img(stitched_img)