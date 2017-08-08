function out = myresize(img, dim, type)
% resize to fixed dimension according to short side or long side
% --------------------------------------------------------------

if nargin < 3
    error('input not enough!');
end
if ~strcmp(type, 'short') && ~strcmp(type, 'long')
    error('wrong type!');
end
H = size(img, 1);
W = size(img, 2);

if strcmp(type, 'short')
    if H <= W
        out = imresize(img, [dim, NaN], 'bilinear');
    else
        out = imresize(img, [NaN, dim], 'bilinear');
    end
else
    if H <= W
        out = imresize(img, [NaN, dim], 'bilinear');
    else
        out = imresize(img, [dim, NaN], 'bilinear');
    end
end
