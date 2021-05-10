import numpy as np
from glob import glob 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 

def read_mat_files(mat_folderpath):
    mat_filepaths = glob(mat_folderpath + '*')
    mat_filepaths = sorted(mat_filepaths)
    print("Total {} .mat files found".format(len(mat_filepaths)))
    #print("Sample filepaths:{}".format(mat_filepaths[:5]))
    img_crops_list = []
    for filepath in mat_filepaths :
        data = loadmat(filepath)
        arr = data['crop_g']
        img_crops_list.append(arr)
    img_crops_arr =  np.array(img_crops_list)
    return img_crops_arr

def get_summed_img_mask(crop_imgs_reshaped, step_h, step_w):
    crop_imgs_reshaped[crop_imgs_reshaped<0] = 0
    num_crops_h, num_crops_w, crop_h, crop_w, crop_c = crop_imgs_reshaped.shape
    img_summed_h = crop_h + (num_crops_h-1)*step_h
    img_summed_w = crop_w + (num_crops_w-1)*step_w
    img_summed_c = crop_c
    img_summed = np.zeros( (img_summed_h , img_summed_w, img_summed_c) , dtype='float32')
    mask       = np.zeros( (img_summed_h , img_summed_w, img_summed_c) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            img_summed[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, : ] += crop_imgs_reshaped[idx_h, idx_w, :,:, :]
            mask[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w, : ] += 1
    return img_summed, mask

def visualize_save_stitched_img(img):
    num_channels = img.shape[2]    
    for i in range(num_channels) :
        plt.figure()
        plt.imshow(img[:,:,i])
        plt.title("stitched_image" + str(i))
        plt.savefig("stitched_{}.jpg".format(i))

def matsave_stitched_img(stitched_img, key_name='crop_g', out_filepath='Actin_2-4-3x5.mat'):
    out_dict = {key_name:stitched_img}
    savemat(out_filepath, out_dict)

mat_folderpath = "D:/NNData/3D/FairSIM3D_082420/test_result/" 
crop_imgs      = read_mat_files(mat_folderpath) 
print("Range of values in crop_imgs: {}-{}".format(np.min(crop_imgs), np.max(crop_imgs)))
print("Shape of img_crops_arr is {}. Dtype is {}".format(crop_imgs.shape, crop_imgs.dtype))
num_crops, crop_h, crop_w, crop_c = crop_imgs.shape

step_h = 64
step_w = 64
stitched_img_h = 1024
stitched_img_w = 1024
num_crops_h = (stitched_img_h - crop_h)//step_h + 1
num_crops_w = (stitched_img_w - crop_w)//step_w + 1
print("Number of crops in height direction: {} , number of crops in width direction: {}".format(num_crops_h, num_crops_w))
    
crop_imgs_reshaped = crop_imgs.reshape(num_crops_h, num_crops_w, crop_h, crop_w, crop_c)
img_summed, mask = get_summed_img_mask(crop_imgs_reshaped, step_h, step_w) 
mask_inv = 1/mask
stitched_img = img_summed * mask_inv
print("Range of values in stitched_img before rounding and before typecasting: {}-{}".format(np.min(stitched_img), np.max(stitched_img)))
print("Shape of stitched image: {}".format(stitched_img.shape))
stitched_img = np.round(stitched_img)
stitched_img = stitched_img.astype('uint8') # converting to dtype as uint8 - same as the dtype of cropped images 
visualize_save_stitched_img(stitched_img)
matsave_stitched_img(stitched_img)
plt.show()



