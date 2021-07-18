from glob import glob 
is_3d = False                                                     
train_model = False
test_model = True
Nthe = 3                                                         
Nphi = 5                                                         
pickle_n = 50                                                     
valid_limit = 1                                                  
use_valid_file = False                                           
div_dataset = False                                              
normalize = True                                                 
size_3rd_dim = 3                                                 
convert_to_2d = False
in_out_same_size = True
out_channels = 1
if train_model and not test_model:
    valid_in = 'D:/NNData/DV_Data/NNData_0626/in/inFairSIM'
    valid_out = 'D:/NNData/DV_Data/NNData_0626/out/outFairSIM'
    pickle_loc = 'D:/NNData/DV_Data/NNData_0626/pickle/'
    model_loc = 'D:/NNData/DV_Data/NNData_0626/model/'
    out_dir = 'D:/NNData/DV_Data/NNData_0626/pred/' #output directory
    inp_fname = 'D:/NNData/DV_Data/NNData_0626/in/inFairSIM'
    out_fname = 'D:/NNData/DV_Data/NNData_0626/out/outFairSIM'
    model_file = "D:/NNData/DV_Data/NNData_0626/model/150.pt"

if test_model and not train_model:
    out_dir = 'D:/NNData/DV_Data/NNData_0626/test_result/' #output directory
    inp_fname ='D:/NNData/DV_Data/NNData_0626/test_valid/in/inFairSIM'
    model_loc = 'D:/NNData/DV_Data/NNData_0626/model/'

save_model_per_epoch = False
save_model_per_interval = True
interval = 100
load_model = False      
mat_files = glob(inp_fname + '*')
mat_files = sorted(mat_files)
limit = len(mat_files)
print("Total {} .mat input files for Test/Train".format(len(mat_files)))