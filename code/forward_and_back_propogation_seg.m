function [target_area, res, seg_result] = forward_and_back_propogation_seg(y, seg_mask_target, net)
% do forward and back-propogation on target pixels in segmentation
% ----------------------------------------------------------------

seg_mask_target(seg_mask_target==0) = 1;

%% this part is the forward propagation process
net.blobs('data').reshape([size(y),1])
net.reshape();
out = net.forward({y}); %do forward pass
out = softmax_dim(out{1}, 3);

[~, seg_result] = max(out, [], 3);


match_situation = (seg_result == seg_mask_target);

match_area = sum(sum(match_situation));
target_area = size(seg_mask_target,1)*size(seg_mask_target,2) - match_area;

seg_mask_target = seg_mask_target.*(1-match_situation);
seg_mask_target(seg_mask_target==0) = 255;

seg_result_target = seg_result.*(1-match_situation);
seg_result_target(seg_result_target==0) = 255;


if isempty(target_area)
    target_area = 0;
    res = 0;
else
    %% this part is for backpropogation process (only do this if there exist target pixels)
    % first consider the gradient to get the target
    dzdy_temp = zeros(net.blobs(net.blob_names{end}).shape, 'single');
    mask_predicition_target_fool = reshape(seg_mask_target, numel(seg_mask_target), 1); 
    dzdy = reshape(dzdy_temp, numel(dzdy_temp), 1);
    
    position = ((mask_predicition_target_fool-1)*length(mask_predicition_target_fool))' + (1:length(mask_predicition_target_fool));
    idx = (mask_predicition_target_fool>0 & mask_predicition_target_fool ~= 255);
    dzdy(position(idx)) = 1;
    
    dzdy = reshape(dzdy, size(out));
    res_fool = net.backward({dzdy}); %do backward pass
    res_fool = res_fool{1};   
    
    % next consider prediciton part
    mask_predicition = reshape(seg_result_target, numel(seg_result_target), 1); 
    dzdy = reshape(dzdy_temp, numel(dzdy_temp), 1);
    
    position = ((mask_predicition-1)*length(mask_predicition))' + (1:length(mask_predicition));
    idx = (mask_predicition>0 & mask_predicition ~= 255);
    dzdy(position(idx)) = 1;
    
    dzdy = reshape(dzdy, size(out));
    res_pred = net.backward({dzdy}); %do backward pass
    res_pred = res_pred{1};
    
    res = res_fool - res_pred;
    
    
end

end

