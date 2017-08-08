function detection_visualization(y, boxes, net, config)
% visualize detection results of the adversarial examples
% -------------------------------------------------------

try
    eval(config);
catch
    keyboard;
end

idx = ones(size(boxes,1),1);
rois = cat(2, idx, boxes);
rois = rois - 1;
rois = permute(rois, [2 1]);

net.blobs('data').reshape([size(y),1])
net.blobs('rois').reshape([size(rois)])
net.reshape();
input_blobs = cell(2, 1);
input_blobs{1} = y;
input_blobs{2} = rois;
scores = net.forward(input_blobs); %do forward pass
scores = scores{1}';
scores = softmax_dim(scores, 2);
scores(:,1) = 0; % set background class to 0
% visualize
boxes_cell = cell(21, 1); % pascal has 21 classes
thres = 0.5; % visualization threshold is 0.5
for i = 1:length(boxes_cell)
    boxes_cell{i} = [boxes, scores(:, i)];
    boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
    
    I = boxes_cell{i}(:, 5) >= thres;
    boxes_cell{i} = boxes_cell{i}(I, :);
end

image = permute(y, [2,1,3]);
image = bsxfun(@plus, image, mean_data);
image = image(:, :, [3,2,1]);

showboxes(uint8(image), boxes_cell, legends, 'voc');

fprintf('this demo use original proposals and  no bbox-regression is applied, please use original faster-rcnn full code to do test!\n')


end

