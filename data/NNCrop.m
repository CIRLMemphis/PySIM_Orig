function NNCrop(g, outX, outY, outZ, Sx, Sy, Sz, str)
[X, Y, Z, ~, ~] = size(g);
cnt = 1;
for xInd = 1:Sx:X-outX+1
    for yInd = 1:Sy:Y-outY+1
        for zInd = 1:Sz:Z-outZ+1
            crop_g        = g(xInd:xInd+outX-1, yInd:yInd+outY-1, zInd:zInd+outZ-1, :, :);
            cnt_strPadded = sprintf( '%06d', cnt ) ;
            save(string(str) + string(cnt_strPadded) + ".mat", 'crop_g');
            cnt = cnt + 1;
        end
    end
end
end