run("C:/Users/cvan/OneDrive - The University of Memphis/CurrentSIM/CIRLSetup.m");

datasets = {"OMX_LSEC_Actin_525nm",...
            "OMX_LSEC_Membrane_680nm", ...
            "OMX_Tetraspeck200_680nm", ...
            "OMX_U2OS_Tubulin_525nm"};
% datasets = {"OMX_LSEC_Actin_525nm"};
bgVals   = [85, 350, 80, 140];
slides   = {4:6; ...
            4:6; ...
            3:5; ...
            3:5};
regionX  = {1:1024;...
            1:1024-128;...
            385:384+384;...
            513:1024-256}; % region of interest in X dimension, -1 for everything
regionY  = {1:1024;...
            129:1024;...
            1:1024;...
            257:1024-256}; % region of interest in Y dimension, -1 for everything

%%
fileCnt = 1;
for ind = 1:length(datasets)
    FileTif      = char(CIRLDataPath + "/SIM_NN/FairSIM/" + datasets(ind) + "_ResFairSIM.tif");
    InfoImage    = imfinfo(FileTif);
    mImage       = InfoImage(1).Width;
    nImage       = InfoImage(1).Height;
    %NumberImages = length(InfoImage);
    reconOb      = zeros(length(regionX{ind}),length(regionY{ind}),length(slides{ind}),'uint16');
    cnt = 1;
    for i = slides{ind}
        temp             = imread(FileTif,'Index',i);
        reconOb(:,:,cnt) = temp(regionX{ind}, regionY{ind});
        cnt = cnt + 1;
    end
    reconOb    = double(reconOb);
    %imshow3D(reconOb); colormap jet;
    %size(reconOb)
    %pause
    endObCnt = NNCrop(reconOb, 128*2, 128*2, 1, 32*2, 32*2, 1, 'out/outFairSIM', fileCnt);
    
    % generate input data
    g = ReadTIF(char(CIRLDataPath + "/SIM_NN/FairSIM/" + datasets(ind) + ".tif"), 3, 5);
    g = g - bgVals(ind);
    g = g(:,:,slides{ind},:,:);
    [Y, X, Z, Nthe, Nphi] = size(g);
    gUp = zeros(2*Y, 2*X, Z, Nthe, Nphi);
    for l = 1:Nthe
        for k = 1:Nphi
            gUp(:,:,:,l,k) = real(IFT(padarray(FT(g(:,:,:,l,k)),[Y/2 X/2 0])));
        end
    end
    gCut = zeros(length(regionX{ind}),length(regionY{ind}),length(slides{ind}));
    for zInd = 1:Z
        for l = 1:Nthe
            for k = 1:Nphi
                temp = fadeBorderCos(gUp(:,:,zInd,l,k), 15*2);
                gCut(:,:,zInd,l,k) = temp(regionX{ind}, regionY{ind});
            end
        end
    end
    endRawCnt = NNCrop(gCut, 128*2, 128*2, 1, 32*2, 32*2, 1, 'in/inFairSIM', fileCnt);
    
    assert(endObCnt == endRawCnt);
    fileCnt = endObCnt;
end