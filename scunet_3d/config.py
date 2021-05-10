### Input & Model Config ###
is_3d = True                                                     ## Disable if 2D before you run train2D.py
limit = 169                                                      ## How many image files we want to train
Nthe = 3                                                         ## Theta
Nphi = 5                                                         ## Phases
normalize = True                                                 ## Apply normalization to set dynamic range from 0 to 1
size_3rd_dim = 3                                                 ## for 3rd dim in 3d data
data_reduced = False
save_mat = False
convert_to_2d = True
reduced_size = Nthe*Nphi* size_3rd_dim
in_out_same_size = False
model_loc = 'D:/NNData/3D/FairSIM3D_042221/model/'
out_dir = 'D:/NNData/3D/FairSIM3D_042221/test_result/'                
inp_fname = 'D:/Final/in3Da/in3DFairSIM'
#out_fname = 'D:/NNData/3D/FairSIM3D_082420/test_valid/out/out3DFairSIM'