### Input & Model Config ###
## Please select if
# (a) is_3d or not (b)'test or train'
# (c) Set Number of input images (Nthe,Nphi) 
# (d) Set limit -No of images to test or train

is_3d = True
train_model = False
test_model = True
data_reduced = False
limit =169
Nthe = 3                                                         
Nphi = 5                                                         
pickle_n = 5                                                     
valid_limit = 1                                                  
use_valid_file = False                                           
div_dataset = False                                              
normalize = True                                                 
size_3rd_dim = 3                                                 
save_mat = False
single_unet = False                                              

## Set train params
save_model_per_epoch = False
save_model_per_interval = True
interval = 50
load_model = False      

if is_3d:
    convert_to_2d = True
    in_out_same_size = False
    x_channel = 208
    if train_model and not test_model: #to avoid errord
        valid_in = 'D:/NNData/3D/FairSIM3D_042221/validation/in/in3DFairSIM'
        valid_out = 'D:/NNData/3D/FairSIM3D_042221/validation/out/out3DFairSIM'
        model_loc = 'D:/NNData/3D/FairSIM3D_042221/model/'
        pickle_loc = 'D:/NNData/3D/FairSIM3D_042221/pickle/'
        out_dir = 'D:/NNData/3D/FairSIM3D_042221/pred/'                
        inp_fname = 'D:/NNData/3D/FairSIM3D_042221/in/in3DFairSIM'
        out_fname = 'D:/NNData/3D/FairSIM3D_042221/out/out3DFairSIM'
        model_file = "D:/NNData/3D/FairSIM3D_042221/model/650.pt"

    if test_model and not train_model:
            model_loc = 'D:/NNData/3D/FairSIM3D_042221/model/'  
            out_dir = 'D:/NNData/3D/FairSIM3D_042221/test_result/'                
            inp_fname = 'D:/Final/in3Da/Slides-5-7/in/in3DFairSIM' 

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
       inp_fname ='D:/NNData/NNData_0626/test_valid/in/inFairSIM'
       model_loc = 'D:/NNData/NNData_0626/model/'
