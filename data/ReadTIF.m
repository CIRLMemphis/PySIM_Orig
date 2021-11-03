function retMat = ReadTIF(filename, Nalp, Nphi)
FileTif      = filename;
InfoImage    = imfinfo(FileTif);
mImage       = InfoImage(1).Width;
nImage       = InfoImage(1).Height;
NumberImages = length(InfoImage);
FinalImage   = zeros(nImage,mImage,NumberImages,'uint16');
for i = 1:NumberImages
   FinalImage(:,:,i) = imread(FileTif,'Index',i);
end

Nz  = NumberImages/Nalp/Nphi;
cnt = 1;
retMat = zeros(nImage, mImage, Nz, Nalp, Nphi);
for j = 1:Nalp
    for k = 1:Nz
        for i = 1:Nphi
            retMat(:,:,k,j,i) = FinalImage(:,:,cnt);
            cnt = cnt + 1;
        end
    end
end
end