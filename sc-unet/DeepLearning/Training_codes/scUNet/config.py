out_dir = 'drive/My Drive/datasets/same/pred/' #output directory
inp_fname = 'drive/My Drive/datasets/same/in/inFairSIM'
out_fname = 'drive/My Drive/datasets/same/out/outFairSIM'
limit = 100 #How many image files we want to train
Nthe = 3
Nphi = 5
div_lr = 'D:/Momin/datasets/LR/'
div_hr = 'D:/Momin/datasets/HR/'
valid_in = 'drive/My Drive/datasets/same/valid/in/inFairSIM'
valid_out = 'drive/My Drive/datasets/same/valid/out/outFairSIM'
pickle_loc = 'drive/My Drive/datasets/same/pickle/'
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


