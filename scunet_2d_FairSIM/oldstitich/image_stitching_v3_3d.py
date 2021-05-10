# importing required libraries
import numpy as np
from glob import glob 
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 

# defining function to read the .mat files from disk and return them as a single numpy array 
def read_mat_files(mat_folderpath):
    mat_filepaths = glob(mat_folderpath + '*')
    mat_filepaths = sorted(mat_filepaths)
    print("Total {} .mat files found".format(len(mat_filepaths)))

    img_crops_list = []
    for filepath in mat_filepaths :
        data = loadmat(filepath)
        #print(data.keys())
        arr = data['crop_g']
        img_crops_list.append(arr)
    #print(arr.dtype, arr.shape) # prints uint16 (256, 256, 3)

    img_crops_arr =  np.array(img_crops_list)
    # print(img_crops_arr.dtype) # prints uint16 

    return img_crops_arr

# defining function to get the summed image 
def get_summed_img(img_crops_arr, step_h, step_w):
    
    num_crops_h, num_crops_w, crop_h, crop_w, crop_channels, num_alpha = img_crops_arr.shape
    img_summed_h = crop_h + (num_crops_h-1)*step_h
    img_summed_w = crop_w + (num_crops_w-1)*step_w

    img_summed = np.zeros( (img_summed_h , img_summed_w, crop_channels, num_alpha) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            img_summed[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w , :, :] += img_crops_arr[idx_h, idx_w,:,:,:,:]
    return img_summed

# defining function to get the multiplication mask for the image  
def get_averaging_mask(img_crops_arr, step_h, step_w):
    
    num_crops_h, num_crops_w, crop_h, crop_w, crop_channels, num_alpha = img_crops_arr.shape
    mask_h = crop_h + (num_crops_h-1)*step_h
    mask_w = crop_w + (num_crops_w-1)*step_w

    mask = np.zeros( (mask_h , mask_w, crop_channels, num_alpha) , dtype='float32')
    for idx_h in range(num_crops_h):
        for idx_w in range(num_crops_w):
            offset_h = idx_h*step_h
            offset_w = idx_w*step_w
            mask[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w , :,:] += 1

    return mask


# defining function for visualizing the masks 
def visualize_masks(mask, mask_inv) :
    plt.figure()
    plt.title("mask")
    plt.imshow(mask[:,:,0,0]/np.max(mask))

    plt.figure()
    plt.imshow(mask_inv[:,:,0,0])
    plt.title("inv mask")
    plt.show()

# # defining function for visualizing and saving the output stitched image  
def visualize_save_stitched_img(img):
    #print(np.min(img), np.max(img), img.dtype)
    for i in range(3):
        for j in range(3):
            plt.figure()
            plt.title("Output stitched image (scaling to 0-1) {},{}".format(i,j))
            img_channel = img[:,:,i,j]
            plt.imshow(img_channel/np.max(img_channel)) 
            plt.savefig("stitched_{}_{}.jpg".format(i,j))

    plt.show()

# defining function for visualizing the cropped image patches 
def visualize_crop_imgs(img_crops_arr):
    plt.figure()
    plt.title("Sample cropped image patch")
    id_h = 0
    id_w = 0
    img_crop = img_crops_arr[id_h, id_w, :,:,0, 0]
    plt.imshow(img_crop/np.max(img_crop)) # np.max(img_crop)
    #plt.show()

# defining function for saving the stitched image in .mat format
def matsave_stitched_img(stitched_img, key_name='crop_g', out_filepath='stitched.mat'):
    for i in range(3):
        for j in range(3):
            out_dict = {key_name:stitched_img[:,:,i,j]}
            savemat(str(i) + '_' + str(j) + '_' + out_filepath, out_dict)


# MAIN CODE BEGINS
# reading mat files into numpy array 
mat_folderpath = "D:/NNData/3D/FairSIM3D_082420/test_data/out/" # update the mat folder location as per your system. use forward slash / on mac/linux systems
img_crops_arr  =  read_mat_files(mat_folderpath)
print("Shape of img_crops_arr is {}. Dtype is {}".format(img_crops_arr.shape, img_crops_arr.dtype))

# defining step_w and step_h parameters corresponding to step sizes in the width and height directions
step_h = 64
step_w = 64
# defining height and width of the stitched image 
stitched_img_h = 1024
stitched_img_w = 1024

# finding various parameters like crop_height, width, number of crops, etc 
num_crops_total, crop_h, crop_w, crop_channels, num_alpha = img_crops_arr.shape
print(num_crops_total, crop_h, crop_w, crop_channels, num_alpha)
num_crops_h = (stitched_img_h - crop_h)//step_h + 1
num_crops_w = (stitched_img_w - crop_w)//step_w + 1
print("Number of crops in height direction: {} , number of crops in width direction: {}".format(num_crops_h, num_crops_w))

# error checking 
assert num_crops_total == num_crops_h * num_crops_w 
assert stitched_img_h  == crop_h + (num_crops_h-1)*step_h
assert stitched_img_w  == crop_w + (num_crops_w-1)*step_w

# reshaping img_crops_arr to shape num_crops_h, num_crops_w, crop_h,crop_w,crop_channels
img_crops_arr = img_crops_arr.reshape(num_crops_h, num_crops_w, crop_h,crop_w,crop_channels, num_alpha)
print("Shape of img_crops_arr after reshaping is {}".format(img_crops_arr.shape))
print(np.min(img_crops_arr), np.max(img_crops_arr))

#visualize_crop_imgs(img_crops_arr)

# computing an output summed image where the small cropped images are copied to their respective locations. values at overlapping regions are summed 
img_summed = get_summed_img(img_crops_arr, step_h, step_w) 
print(img_summed.shape, img_summed.dtype)

# computing the averaging mask for perfoming averaging in the overlapping regions 
mask = get_averaging_mask(img_crops_arr, step_h, step_w)
print(mask.shape, mask.dtype)
mask_inv = 1/mask  # for averaging
print(mask_inv.shape, mask_inv.dtype)

# visualizing mask and inverse masks
#visualize_masks(mask, mask_inv)

# generating stitched image
stitched_img = img_summed * mask_inv
stitched_img = np.round(stitched_img) # rounding to nearest integer value 
stitched_img = stitched_img.astype('uint16') # converting to dtype as uint16 - same as the dtype of cropped images 
print(np.min(stitched_img), np.max(stitched_img), stitched_img.dtype)

# visualizing and saving the output stitched image 
visualize_save_stitched_img(stitched_img)

# save stitched image in .mat format
matsave_stitched_img(stitched_img)

# showing the plots previously created 
plt.show()