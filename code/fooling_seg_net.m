function [r, itr, status, box_num, seg_result] = fooling_seg_net(x, seg_mask_target, seg_mask_ori, net, config)

try
    eval(config);
catch
    keyboard;
end

% initilization of fooling process
r = x * 0;
itr = 0;

% intilization of fooling target
[pred_target, dr, seg_result] = forward_and_back_propogation_seg(x+r, seg_mask_target, net); %generate qualified bbox for back-propagation (e.g., how many boxes are said it is a car)
box_num(itr+1) = pred_target;

pred_target_ori = sum(sum(seg_mask_ori>0));

while (pred_target > 0.01*pred_target_ori) && itr<MAX_ITER % do not need pred_target
    itr = itr + 1;
    sprintf('iteration number %d\n', itr)
    
    % process of noise
    dr_temp = reshape(dr, numel(dr), 1);
    r_gain = step_length/max(abs(dr_temp));
    r = r + dr*r_gain;
    r_max = max(reshape(abs(r), numel(r), 1));
    fprintf('max value in the perturbation is %.2f\n', r_max);
    
    % calculate the candidate for the next interation
    [pred_target, dr, seg_result] = forward_and_back_propogation_seg(x+r, seg_mask_target, net); %generate qualified bbox for back-propagation (e.g., how many boxes are said it is a car)
    fprintf('%d pixels remained\n', pred_target);
    box_num(itr+1) = pred_target;
end

if pred_target >= 0.01*pred_target_ori
    status = 0;
else
    status = 1;
end

end