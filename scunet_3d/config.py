### Input & Model Config ###
is_3d = True                                                     
## Please choose if (a) 'test or train'; (b) Set Number of input images (Nthe,Nphi) (c) Set limit -No of images to test or train
## To stitch set folder accordingly and run 3d_stitich_3d_data.py
train_model = False
test_model = True
limit = 169                                                      ## How many image files we want to train
Nthe = 3                                                         ## Theta
Nphi = 5                                                         ## Phases
pickle_n = 5                                                     ## Model saved limit
valid_limit = 1                                                  ## Validation saved limit
use_valid_file = False                                           ## Validate simultaneously ?
div_dataset = False                                              ## Txfer learn dataset
normalize = True                                                 ## Apply normalization to set dynamic range from 0 to 1
size_3rd_dim = 3                                                 ## for 3rd dim in 3d data
data_reduced = False
save_mat = False
single_unet = False                                              ## Don't need this really but its what they do
save_model_per_epoch = False
save_model_per_interval = True
convert_to_2d = True
reduced_size = Nthe*Nphi* size_3rd_dim
in_out_same_size = False        

if train_model and not test_model:
    valid_in = 'D:/NNData/3D/FairSIM3D_042221/validation/in/in3DFairSIM'
    valid_out = 'D:/NNData/3D/FairSIM3D_042221/validation/out/out3DFairSIM'
    model_loc = 'D:/NNData/3D/FairSIM3D_042221/model/'
    pickle_loc = 'D:/NNData/3D/FairSIM3D_042221/pickle/'
    out_dir = 'D:/NNData/3D/FairSIM3D_042221/pred/'                
    inp_fname = 'D:/NNData/3D/FairSIM3D_042221/in/in3DFairSIM'
    out_fname = 'D:/NNData/3D/FairSIM3D_042221/out/out3DFairSIM'
    load_model = False
    model_file = "D:/NNData/3D/FairSIM3D_042221/model/650.pt"
    interval = 100

if test_model and not train_model:
    model_loc = 'D:/NNData/3D/FairSIM3D_042221/model/'  
    out_dir = 'D:/NNData/3D/FairSIM3D_042221/test_result/'                
    inp_fname = 'D:/Final/in3Da/Slides-5-7/in/in3DFairSIM' 