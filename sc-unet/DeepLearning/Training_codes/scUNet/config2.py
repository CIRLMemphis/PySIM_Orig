out_dir = 'D:/NNData/simulated/NNSimulatedData_080820/pred/' #output directory
inp_fname = 'D:/NNData/simulated/NNSimulatedData_080820/in/inSim'
out_fname = 'D:/NNData/simulated/NNSimulatedData_080820/out/outSim'
limit = 1000 #How many image files we want to train
Nthe = 3
Nphi = 5
div_lr = 'D:/Momin/datasets/LR/'
div_hr = 'D:/Momin/datasets/HR/'
valid_in = 'D:/NNData/simulated/NNSimulatedData_080820/validation/in/inSim'
valid_out = 'D:/NNData/simulated/NNSimulatedData_080820/validation/out/outSim'
pickle_loc = 'D:/NNData/simulated/NNSimulatedData_080820/pickle/'
pickle_n = 50
valid_limit = 1
use_valid_file = True
div_dataset = False
in_channels = 15
out_channels = 1
min_lr = .00001
lr_c = 150000
active_inputs = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
in_channels = sum(active_inputs)
is_reduced = False
reduced_size = 7
in_out_same_size = False



out_dir = 'D:/NNData/NNData_0520/pred/' #output directory
inp_fname = 'D:/NNData/NNData_0520/in/inFairSIM'
out_fname = 'D:/NNData/NNData_0520/out/outFairSIM'
limit = 1000 #How many image files we want to train
Nthe = 3
Nphi = 5
div_lr = 'D:/Momin/datasets/LR/'
div_hr = 'D:/Momin/datasets/HR/'
valid_in = 'D:/NNData/NNData_0520/valid/in/inFairSIM'
valid_out = 'D:/NNData/NNData_0520/valid/out/outFairSIM'
pickle_loc = 'D:/NNData/NNData_0520/pickle/'
pickle_n = 50
valid_limit = 1
use_valid_file = True
div_dataset = False
in_channels = 15
out_channels = 1
min_lr = .00001
lr_c = 150000
active_inputs = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
in_channels = sum(active_inputs)
is_reduced = False
reduced_size = 7
in_out_same_size = False
normalize = False
is_3d = False
size_3rd_dim = 1





out_dir = 'D:/NNData/3D/FairSIM3D_082420/pred/' #output directory
inp_fname = 'D:/NNData/3D/FairSIM3D_082420/in/in3DFairSIM'
out_fname = 'D:/NNData/3D/FairSIM3D_082420/out/out3DFairSIM'
limit = 330 #How many image files we want to train
Nthe = 3
Nphi = 5
div_lr = 'D:/Momin/datasets/LR/'
div_hr = 'D:/Momin/datasets/HR/'
valid_in = 'D:/NNData/3D/FairSIM3D_082420/validation/in/in3DFairSIM'
valid_out = 'D:/NNData/3D/FairSIM3D_082420/validation/out/out3DFairSIM'
pickle_loc = 'D:/NNData/3D/FairSIM3D_082420/pickle/'
pickle_n = 50
valid_limit = 1
use_valid_file = True
div_dataset = False
in_channels = 15
out_channels = 1
min_lr = .00001
lr_c = 150000
active_inputs = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
in_channels = sum(active_inputs)
is_reduced = False
reduced_size = 7
in_out_same_size = True
normalize = False
is_3d = True
size_3rd_dim = 3
convert_to_2d = True

