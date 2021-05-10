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
    img_crops_arr =  np.array(img_crops_list)
    return img_crops_arr

def get_summed_img(img_crops_arr, step_h, step_w):
    num_crops_h, num_crops_w, crop_h, crop_w = img_crops_arr.shape
    img_summed_h = crop_h + (num_crops_h-1)*step_h
    img_summed_w = crop_w + (num_crops_w-1)*step_w
    img_summed = np.zeros( (img_summed_h , img_summed_w) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            img_summed[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w ] += img_crops_arr[idx_h, idx_w,:,:]
    return img_summed

def get_averaging_mask(img_crops_arr, step_h, step_w):
    num_crops_h, num_crops_w, crop_h, crop_w = img_crops_arr.shape
    mask_h = crop_h + (num_crops_h-1)*step_h
    mask_w = crop_w + (num_crops_w-1)*step_w
    mask = np.zeros( (mask_h , mask_w) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            mask[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w ] += 1
    return mask

def visualize_save_stitched_img(img, i):
    plt.figure()
    plt.title("Output stitched image (scaling to 0-1)")
    plt.imshow(img/np.max(img)) 
    plt.savefig("stitched_{}.jpg".format(i))

def matsave_stitched_img(stitched_img, key_name='crop_g', out_filepath='Actin_2-4-3x1.mat'):
    out_dict = {key_name:stitched_img}
    savemat(out_filepath, out_dict)


mat_folderpath = "D:/NNData/NNData_0626/test_result/" 
all_imgs_crops_arr  =  read_mat_files(mat_folderpath)
print("Shape of img_crops_arr is {}. Dtype is {}".format(all_imgs_crops_arr.shape, all_imgs_crops_arr.dtype))
step_h = 64
step_w = 64
stitched_img_h = 1024
stitched_img_w = 1024
num_output_images = 3 
total_num_crops = all_imgs_crops_arr.shape[0]
num_crops_per_image = total_num_crops // num_output_images
output_stitched_images_list = []
for i in range(num_output_images) :
    print("Processing {} image".format(i+1))
    #start_idx = i*num_crops_per_image
    #end_idx   = (i+1)*num_crops_per_image
    img_crops_arr = all_imgs_crops_arr[i: total_num_crops-(2-i) : num_output_images]
    num_crops_total, crop_h, crop_w  = img_crops_arr.shape
    num_crops_h = (stitched_img_h - crop_h)//step_h + 1
    num_crops_w = (stitched_img_w - crop_w)//step_w + 1
    print("Number of crops in height direction: {} , number of crops in width direction: {}".format(num_crops_h, num_crops_w))
    # error checking like in Dr Van
    assert num_crops_total == num_crops_h * num_crops_w 
    assert stitched_img_h  == crop_h + (num_crops_h-1)*step_h
    assert stitched_img_w  == crop_w + (num_crops_w-1)*step_w
    img_crops_arr = img_crops_arr.reshape(num_crops_h, num_crops_w, crop_h,crop_w)
    img_summed = get_summed_img(img_crops_arr, step_h, step_w) 
    mask = get_averaging_mask(img_crops_arr, step_h, step_w)
    mask_inv = 1/mask
    stitched_img = img_summed * mask_inv
    stitched_img = np.round(stitched_img)
    stitched_img = stitched_img.astype('uint16') # converting to dtype as uint16 - same as the dtype of cropped images 
    stitched_img = np.squeeze(stitched_img)
    visualize_save_stitched_img(stitched_img,i)
    output_stitched_images_list.append(stitched_img)
all_stitched_images = np.stack(output_stitched_images_list, axis=-1)
print(all_stitched_images.shape)
matsave_stitched_img(all_stitched_images)
plt.show()