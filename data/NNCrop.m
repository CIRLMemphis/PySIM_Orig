function cnt = NNCrop(g, outX, outY, outZ, Sx, Sy, Sz, str, startCnt)
[X, Y, Z, ~, ~] = size(g);
if (nargin < 9 || startCnt == -1)
    cnt = 1;
else
    cnt = startCnt;
end
for xInd = 1:Sx:X-outX+1
    for yInd = 1:Sy:Y-outY+1
        for zInd = 1:Sz:Z-outZ+1
            crop_g        = g(xInd:xInd+outX-1, yInd:yInd+outY-1, zInd:zInd+outZ-1, :, :);
            cnt_strPadded = sprintf( '%06d', cnt ) ;
%             f = figure('visible', 'off');
%             imagesc(crop_g(:,:,1,1,1)); axis square;
%             print('-djpeg', char(string(str) + string(cnt_strPadded) + ".jpg"));
%             close(f)
            save(string(str) + string(cnt_strPadded) + ".mat", 'crop_g');
            cnt = cnt + 1;
        end
    end
end
cnt - 1
end