### Input & Model Config ###
## Please select if
# (a) is_3d or not (b)'test or train'
# (c) Set Number of input images (Nthe,Nphi) 
# (d) Set limit -No of images to test or train
is_3d = True
train_model = True
test_model = False
data_reduced = False
from glob import glob 
Nthe = 3                                                         
Nphi = 5                                                        
pickle_n = 50                                                     
valid_limit = 1                                                  
use_valid_file = False                                           
div_dataset = False                                              
normalize = True                                                 
size_3rd_dim = 3                                                 
save_mat = False
single_unet = False                                              

## Set train params
if train_model:
    save_model_per_epoch = False
    save_model_per_interval = True
    interval = 100
    load_model = False

if is_3d:
    convert_to_2d = True
    in_out_same_size = False
    x_channel = 64
    if train_model and not test_model:
        valid_in = 'D:/NNData/DV_Data/FairSIM3D_042221/validation/in/in3DFairSIM'
        valid_out = 'D:/NNData/DV_Data/FairSIM3D_042221/validation/out/out3DFairSIM'
        model_loc = 'D:/NNData/DV_Data/FairSIM3D_042221/model/'
        pickle_loc = 'D:/NNData/DV_Data/FairSIM3D_042221/pickle/'
        out_dir = 'D:/NNData/DV_Data/FairSIM3D_042221/pred/'                
        inp_fname = 'D:/NNData/DV_Data/FairSIM3D_042221/in/in3DFairSIM'
        out_fname = 'D:/NNData/DV_Data/FairSIM3D_042221/out/out3DFairSIM'
        model_file = "D:/NNData/DV_Data/FairSIM3D_042221/model/550.pt"

    if test_model and not train_model:
            model_loc = 'D:/NNData/DV_Data/FairSIM3D_042221/model/'  
            out_dir = 'D:/NNData/DV_Data/FairSIM3D_042221/test_result/'                
            inp_fname = 'D:/Final/in3Da/in3DFairSIM' 

## For 2D Data
if not is_3d:
    x_channel = 192
    convert_to_2d = False
    in_out_same_size = True
    if train_model and not test_model:
        valid_in = 'D:/NNData/DV_Data/NNData_0626/in/inFairSIM'
        valid_out = 'D:/NNData/DV_Data/NNData_0626/out/outFairSIM'
        pickle_loc = 'D:/NNData/DV_Data/NNData_0626/pickle/'
        model_loc = 'D:/NNData/DV_Data/NNData_0626/model/'
        out_dir = 'D:/NNData/DV_Data/NNData_0626/pred/' #output directory
        inp_fname = 'D:/NNData/DV_Data/NNData_0626/in/inFairSIM'
        out_fname = 'D:/NNData/DV_Data/NNData_0626/out/outFairSIM'
        model_file = "D:/NNData/DV_Data/NNData_0626/model/200.pt"

    if test_model and not train_model:
       out_dir = 'D:/NNData/DV_Data/NNData_0626/test_result/' #output directory
       inp_fname ='D:/NNData/DV_Data/NNData_0626/test_valid/in/inFairSIM'
       model_loc = 'D:/NNData/DV_Data/NNData_0626/model/'

mat_files = glob(inp_fname + '*')
mat_files = sorted(mat_files)
limit = len(mat_files)
if train_model: 
    print("Total {}.mat input files for Training model".format(len(mat_files)))
else:
    print("Total {}.mat input files for Testing model".format(len(mat_files)))