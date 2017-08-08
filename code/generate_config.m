%% config files of the demo.m

addpath('../data/');
addpath('../functions/');
load('pascal_seg_colormap.mat');
load('legend_voc.mat');

% model_select = 'seg_fcn_8s';
model_select = 'seg_fcn_alexnet';
% model_select = 'det_VGG';
% model_select = 'det_ZF';

if strfind(model_select, 'det')
    MAX_ITER = 150; % max iteration number for detection
    im_name = '2007_000925';
else
    MAX_ITER = 200; % max iteration number for segmentation
    im_name = '2011_003271';
    % choose the geometric shape that you want, e.g. square, circle, strip
    shape = 'square';
end

step_length = 0.5; % the step length of back-propagation direction

%% choose model weight
switch model_select
    case 'seg_fcn_8s'
        net_model = '../prototxt/fcn_8s_deploy.prototxt';
        net_weights = '../weight/fcn8s-heavy-pascal.caffemodel';
    case 'seg_fcn_alexnet'
        net_model = '../prototxt/fcn_alexnet_deploy.prototxt';
        net_weights = '../weight/fcn-alexnet-pascal.caffemodel';
    case 'det_VGG'
        net_model = '../prototxt/faster_detection_VGG.prototxt';
        net_weights = '../weight/detection_final_VGG';
    case 'det_ZF'
        net_model = '../prototxt/faster_detection_ZF.prototxt';
        net_weights = '../weight/detection_final_ZF';
    otherwise
        error('this model is not included in our scripts!')
end

% check if all the models exist
if ~exist(net_weights, 'file')
    run ../fetch_data/fetch_all_models.m
end

%% caffe basic setting, mean data, already in BRG
mean_data(:,:,1) = 103.9390;
mean_data(:,:,2) = 116.7790;
mean_data(:,:,3) = 123.6800;
