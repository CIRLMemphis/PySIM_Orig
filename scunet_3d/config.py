is_3d = True
train_model = True
test_model = False
from glob import glob 
Nthe = 3                                                         
Nphi = 5                                                        
pickle_n = 50                                                     
valid_limit = 1                                                  
use_valid_file = False                                           
normalize = True                                                 
size_3rd_dim = 3                                                 
convert_to_2d = False                                             
if train_model and not test_model:
    save_model_per_epoch = False
    save_model_per_interval = True
    interval = 100
    load_model = False
    in_out_same_size = False
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
mat_files = glob(inp_fname + '*')
mat_files = sorted(mat_files)
limit = len(mat_files)

if train_model:
    print("Total {}.mat input files for Training model".format(len(mat_files)))
else:
    print("Total {}.mat input files for Testing model".format(len(mat_files)))