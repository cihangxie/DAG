function image = image_clip(image)
%IMAGE_BOUND Summary of this function goes here
%   Detailed explanation goes here
image(image>255) = 255;
image(image<0) = 0;

end

