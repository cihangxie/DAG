% downloading all models

addpath('../functions/');

mkdir_if_missing('../weight/');
mkdir_if_missing('./temp');

%% for detection files
fprintf('Downloading faster_rcnn_final_model...\n');
urlwrite('https://onedrive.live.com/download?resid=D7AF52BADBA8A4BC!114&authkey=!AERHoxZ-iAx_j34&ithint=file%2czip', ...
    './temp/faster_rcnn_final_model.zip');

fprintf('Unzipping...\n');
unzip('./temp/faster_rcnn_final_model.zip', './temp/');

movefile ./temp/output/faster_rcnn_final/faster_rcnn_VOC0712_vgg_16layers/detection_final  ../weight/detection_final_VGG
movefile ./temp/output/faster_rcnn_final/faster_rcnn_VOC0712_ZF/detection_final  ../weight/detection_final_ZF
fprintf('Detection Models Done.\n');

rmdir('./temp/', 's');

%% for segmentation files
fprintf('Downloading fcn models...\n')
urlwrite('http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel', ...
    '../weight/fcn8s-heavy-pascal.caffemodel');

urlwrite('http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel', ...
    '../weight/fcn-alexnet-pascal.caffemodel');

fprintf('Segmentation Models Done.\n');


