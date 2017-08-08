function box_label = forward_propogation_det(y, rois, box_label, net)
% do forward-propoagation on all pre-computed bbox
% --------------------------------------

net.blobs('data').reshape([size(y),1])
net.blobs('rois').reshape([size(rois)])
net.reshape();
input_blobs = cell(2, 1);
input_blobs{1} = y;
input_blobs{2} = rois;
out = net.forward(input_blobs); %do forward pass

out = softmax_dim(out{1}, 1);
% do the prediction of strong version
[score_value, box_label(3,:)] = max(out, [], 1);

% enforce strong adversarial result
score_threshold = 0.8;
strong_index = (box_label(3,:) == box_label(2,:)) & (box_label(1,:)~= 1);
if sum(strong_index == 1)
    % use nms to judge it, if works, then do not mind it
    strong_position_ori = find(strong_index==1);
    bbox = cat(1, rois(2:5, strong_position_ori), score_value(strong_position_ori));
    bbox = bbox';
    nms_index = nms(bbox, 0.35);
    bbox = bbox(nms_index, :);
    if any(bbox(:, end) < score_threshold)
        strong_position_ori = strong_position_ori(nms_index);
        idx = bbox(:,end)<score_threshold;
        box_label(3,strong_position_ori(idx)) = box_label(1,strong_position_ori(idx));
        
    end
end

end

