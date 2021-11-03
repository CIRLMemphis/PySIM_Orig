function img = fadeBorderCos(img, px)
[w,h] = size(img);
img = reshape(img, w*h, 1);
fac = 1/px*pi/2;
for y = 0:px-1
    for x = 0:w-1
        img((x+y*w)+1) = img((x+y*w)+1)*sin(y*fac)^2;
    end
end

for y = h-px:h-1
    for x = 0:w-1
        img((x+y*w)+1) = img((x+y*w)+1)*sin((y-h-1)*fac)^2;
    end
end

for y = 0:h-1
    for x = 0:px-1
        img((x+y*w)+1) = img((x+y*w)+1)*sin(x*fac)^2;
    end
end

for y = 0:h-1
    for x = w-px:w-1
        img((x+y*w) + 1) = img((x+y*w) + 1)*sin((w-x-1)*fac)^2;
    end
end

img = reshape(img, w, h);
end