import numpy as np
from glob import glob 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 
slide = 'slide47.mat'
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

def get_summed_img(img_crops_arr, step_h, step_w):
    num_crops_h, num_crops_w, crop_h, crop_w, crop_channels = img_crops_arr.shape
    img_summed_h = crop_h + (num_crops_h-1)*step_h
    img_summed_w = crop_w + (num_crops_w-1)*step_w
    img_summed = np.zeros( (img_summed_h , img_summed_w, crop_channels) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            img_summed[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w , :] += img_crops_arr[idx_h, idx_w,:,:,:]
    return img_summed

def get_averaging_mask(img_crops_arr, step_h, step_w):
    num_crops_h, num_crops_w, crop_h, crop_w, crop_channels = img_crops_arr.shape
    mask_h = crop_h + (num_crops_h-1)*step_h
    mask_w = crop_w + (num_crops_w-1)*step_w
    mask = np.zeros( (mask_h , mask_w, crop_channels) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            mask[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w , :] += 1
    return mask

def visualize_save_stitched_img(img):
    plt.figure()
    plt.title("Output stitched image (scaling to 0-1)")
    plt.imshow(img/np.max(img)) 
    plt.savefig("stitched1.jpg")

def matsave_stitched_img(stitched_img, key_name='crop_g', out_filepath=slide):
    out_dict = {key_name:stitched_img}
    savemat(out_filepath, out_dict)
mat_folderpath = "D:/NNData/NNData_0626/test_result/" # update the mat folder location as per your system. use forward slash / on mac/linux systems
img_crops_arr  =  read_mat_files(mat_folderpath)
print("See shape of img_crops_arr ==>> {}. Dtype is {}".format(img_crops_arr.shape, img_crops_arr.dtype))
step_h = 64
step_w = 64
stitched_img_h = 1024
stitched_img_w = 1024
if len(img_crops_arr.shape) == 3 :   #use momil
    img_crops_arr = img_crops_arr.reshape(*img_crops_arr.shape, 1)
num_crops_total, crop_h, crop_w, crop_channels = img_crops_arr.shape
num_crops_h = (stitched_img_h - crop_h)//step_h + 1
num_crops_w = (stitched_img_w - crop_w)//step_w + 1
img_crops_arr = img_crops_arr.reshape(num_crops_h, num_crops_w, crop_h,crop_w,crop_channels)
img_summed = get_summed_img(img_crops_arr, step_h, step_w) 
mask = get_averaging_mask(img_crops_arr, step_h, step_w)
mask_inv = 1/mask  # for averaging
stitched_img = img_summed * mask_inv
stitched_img = np.round(stitched_img) 
stitched_img = stitched_img.astype('uint16') 
stitched_img = np.squeeze(stitched_img)
#visualize_save_stitched_img(stitched_img)
matsave_stitched_img(stitched_img)
plt.show()