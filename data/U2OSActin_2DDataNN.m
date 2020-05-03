%% input data generation
load(CIRLDataPath + "/FairSimData/OMX_U2OS_Actin_525nm.mat");
size(g)
NNCrop(g, 128, 128, 1, 32, 32, 1, 'U2OSActin')

%% output data generation
load('C:\Users\cvan\OneDrive - The University of Memphis\CIRLData\Results\U2OSActin\202002221320_Exp3WU2OSActinPSFVzMBPC\202002221320_Exp3WU2OSActinPSFVzMBPC.mat', 'retVars')
reconOb = retVars{10};
reconOb(reconOb < 0) = 0;
g = reconOb;
NNCrop(g, 128*2, 128*2, 1, 32*2, 32*2, 2, 'outU2OSActin')