function image = image_clip(image)
% clip images into valid range
% ----------------------------

image(image>255) = 255;
image(image<0) = 0;

end

