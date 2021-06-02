### Input & Model Config ###
## Please choose if
# (a) is_3d or not (b)'test or train'
# (c) Set Number of input images (Nthe,Nphi) 
## To stitch set folder accordingly and run 3d_stitich_(2d/3d)_data.py
from glob import glob 
is_3d = False                                                     
train_model = False
test_model = True
limit = 1182  
Nthe = 3                                                         
Nphi = 5                                                         
pickle_n = 5                                                     
valid_limit = 1                                                  
use_valid_file = False                                           
div_dataset = False                                              
normalize = True                                                 
size_3rd_dim = 3                                                 
data_reduced = False
save_mat = False
single_unet = False                                              

## Set train params
if train_model:
    save_model_per_epoch = False
    save_model_per_interval = True
    interval = 100
    load_model = False      

## For 2D Data
if not is_3d:
    x_channel = 64
    convert_to_2d = False
    in_out_same_size = True
    if train_model and not test_model:
        valid_in = 'D:/NNData/NNData_0626/in/inFairSIM'
        valid_out = 'D:/NNData/NNData_0626/out/outFairSIM'
        pickle_loc = 'D:/NNData/NNData_0626/pickle/'
        model_loc = 'D:/NNData/NNData_0626/model/'
        out_dir = 'D:/NNData/NNData_0626/pred/' #output directory
        inp_fname = 'D:/NNData/NNData_0626/in/inFairSIM'
        out_fname = 'D:/NNData/NNData_0626/out/outFairSIM'
        model_file = "D:/NNData/3D/FairSIM3D_042221/model/650.pt"

    if test_model and not train_model:
       out_dir = 'D:/NNData/NNData_0626/test_result/' #output directory
       inp_fname ='D:/NNData/NNData_0626/inland/in/inFairSIM'
       model_loc = 'D:/NNData/NNData_0626/model/'

mat_files = glob(inp_fname + '*')
mat_files = sorted(mat_files)
limit = len(mat_files)
print("Total {} .mat input files for ops".format(len(mat_files)))