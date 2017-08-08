% % ------------ This script is for generating adversarial examples for
% segmentation and detection (https://arxiv.org/abs/1703.08603)

clear
close all;

config ='generate_config';

try
    eval(config);
catch
    keyboard;
end

% add matlab caffe path
addpath('../caffe/matlab/');

% load caffe models
caffe.set_mode_gpu(); %caffe.set_mode_cpu();
caffe.set_device(0)
caffe.reset_all();
net = caffe.Net(net_model, net_weights, 'test');

%% generating adversarial examples now
fprintf('now generating adversarial examples for %s\n\n', model_select);

% prepare image info
image = imread(sprintf('../data/%s.jpg', im_name));

if strfind(model_select, 'det')
    % for the detection network, input short size is 600
    image = myresize(image, 600, 'short'); 
end
if size(image, 3) == 1
    image = cat(3, image, image, image);
end
% convert image format
image = image(:, :, [3, 2, 1]);  % format: W x H x C with BGR channels
image = single(image);  % convert from uint8 to single
image = bsxfun(@minus, image, mean_data);% subtract mean_data (already in W x H x C, BGR)
image = permute(image, [2, 1, 3]);  % flip width and height

% r means noise perturbation, itr means iteration number
if strfind(model_select, 'det')
    xml_info = VOCreadrecxml(sprintf('../data/%s.xml', im_name));
    annotation = xml_info.objects; % object detection annotation
    ratio = 600/min(xml_info.imgsize(1:2));  
    % use nms = 0.9 in RPN and choose Top 3000 boxes
    load(sprintf('../data/%s_box_3000_%s.mat', im_name, model_select)); 
    boxes = aboxes(:,1:4);
    % extract gt, construct sturcture like [obj_index, bbox]
    gt = zeros(numel(annotation), 5); 
    for j = 1:numel(annotation)
        obj_idx = strfind(legends, annotation(j).class);
        obj_idx = cellfun('isempty', obj_idx);
        obj_idx = find(obj_idx==0);
        gt(j,:) = [obj_idx, ratio*annotation(j).bbox];
    end
    mapping = generate_mapping(unique(gt(:,1))-1);
    mapping(mapping~=0) = mapping(mapping~=0) + 1;
    mapping = [1, mapping]; % leave background class untouched
    [r, itr, status, box_num] = fooling_det_net(image, boxes, gt, net, mapping, config);    
    detection_visualization(image+r, boxes, net, config);
    
else if strfind(model_select, 'seg')
        % prepare segmentation data
        seg_mask_ori = imread(sprintf('../data/%s.png', im_name)); 
        seg_mask_ori(seg_mask_ori == 255) = 0; % ignore white space
        gt_idx = unique(seg_mask_ori);
        gt_idx(gt_idx == 0) = []; % ignore class background
        [~, target_idx_candidate_shuffle] = generate_mapping(gt_idx);        
        load(sprintf('../data/%s.mat', shape)); % load pre-defined mask
        mask(mask~=0) = target_idx_candidate_shuffle(mask(mask~=0)); % assign a random color
        [r, itr, status, box_num, seg_result] = fooling_seg_net(image, double(mask'), double(seg_mask_ori'), net, config);
        imshow(seg_result, colormap);
        
    else
        error('this model type is not available in our setting')
        
    end
end

%% show another visualization
% restore the images to normal status
image_fool = image + r;
image_fool = permute(image_fool, [2,1,3]);
image_fool = bsxfun(@plus, image_fool, mean_data);
image_fool = image_fool(:, :, [3,2,1]);

% also do processing for r
r = permute(r, [2,1,3]);
r = r(:, :, [3,2,1]);

fig = figure(2);
scr_size = get(0,'screensize');
set(fig,'pos',[scr_size(3)/2,scr_size(4)/2,900,250]);
pbaspect([1,1,1])
subplot(1,3,1)
imagesc((image_fool - r)/255)
subplot(1,3,2)
imagesc((image_fool)/255)
subplot(1,3,3)
imagesc(r)

caffe.reset_all();
