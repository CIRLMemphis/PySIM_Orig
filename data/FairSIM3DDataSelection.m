run("C:/Users/cvan/OneDrive - The University of Memphis/CurrentSIM/CIRLSetup.m");

%datasets = {"OMX_LSEC_Actin_525nm",...
%            "OMX_LSEC_Membrane_680nm", ...
%            "OMX_Tetraspeck200_680nm", ...
%            "OMX_U2OS_Tubulin_525nm"};
datasets = {"OMX_LSEC_Actin_525nm", "OMX_LSEC_Membrane_680nm"};
bgVals   = [85, 350, 80, 140];
% slides   = {4:6; ...
%             4:6; ...
%             3:5; ...
%             3:5};
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
endObCnt  = 1;
endRawCnt = 1;
for ind = 1:length(datasets)
    FileTif      = char(CIRLDataPath + "/SIM_NN/FairSIM/" + datasets(ind) + "_ResFairSIM.tif");
    InfoImage    = imfinfo(FileTif);
    mImage       = InfoImage(1).Width;
    nImage       = InfoImage(1).Height;
    %NumberImages = length(InfoImage);
    for slideInd = slides{ind}
        reconOb      = zeros(length(regionX{ind}),length(regionY{ind}),3,'uint16');
        
            temp             = imread(FileTif,'Index',slideInd);
            reconOb(:,:,2) = temp(regionX{ind}, regionY{ind});
            temp             = imread(FileTif,'Index',slideInd-1);
            reconOb(:,:,1) = temp(regionX{ind}, regionY{ind});
            temp             = imread(FileTif,'Index',slideInd+1);
            reconOb(:,:,3) = temp(regionX{ind}, regionY{ind});
        
        reconOb    = double(reconOb);
        %imshow3D(reconOb); colormap jet;
        %size(reconOb)
        %pause
        endObCnt = NNCrop(reconOb, 128*2, 128*2, 3, 32*2, 32*2, 1, 'out/out3DFairSIM', endObCnt);
    end
    
    % generate input data
    g = ReadTIF(char(CIRLDataPath + "/SIM_NN/FairSIM/" + datasets(ind) + ".tif"), 3, 5);
    g = g - bgVals(ind);
    for slideInd = slides{ind}
        gIn = g(:,:,(slideInd-1):(slideInd+1),:,:);
        [Y, X, Z, Nthe, Nphi] = size(gIn);
        gCut = zeros(length(round(regionX{ind}(1:2:end)/2)),...
                     length(round(regionY{ind}(1:2:end)/2)),...
                     3);
        for zInd = 1:Z
            for l = 1:Nthe
                for k = 1:Nphi
                    temp = fadeBorderCos(gIn(:,:,zInd,l,k), 15);
                    gCut(:,:,zInd,l,k) = temp(round(regionX{ind}(1:2:end)/2), round(regionY{ind}(1:2:end)/2));
                end
            end
        end
        endRawCnt = NNCrop(gCut, 128, 128, 3, 32, 32, 1, 'in/in3DFairSIM', endRawCnt);
    end
    
    assert(endObCnt == endRawCnt);
    fileCnt = endObCnt;
end