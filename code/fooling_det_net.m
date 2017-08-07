function [r, itr, status, box_num] = fooling_det_net(x, boxes, gt, net, mapping, config)

try
    eval(config);
catch
    keyboard;
end

% initilization of fooling process
r = x * 0;
itr = 0;

% constrcuct the rois structure
idx = ones(size(boxes,1),1);
rois = cat(2, idx, boxes);
rois = rois - 1;
rois = permute(rois, [2 1]);

% intilization of fooling target
% mapping = 1:21;
box_label = assign_target_det(x, rois, gt, mapping, net); % generate qualified bbox for back-propagation (e.g., how many boxes are said it is a car)
box_num(itr+1) = sum(box_label(1,:) == box_label(3,:) & box_label(1,:) ~= 1);

% first condition is that we need to transfer all related boxes into fooling target
% while (any(box_label(2,:)== box_label(3,:))) && itr<MAX_ITER
while (sum(box_label(1,:) == box_label(3,:) & box_label(1,:) ~= 1)) && itr<MAX_ITER
    
    itr = itr + 1;
    
    fprintf('iteration number %d\n', itr);
    % do the back-propogation for the selected bbox candidates, know the direction to go
    dr = back_propogation_det(x+r, rois, box_label, net);
    
    % process of noise
    dr_temp = reshape(dr, numel(dr), 1);
    r_gain = step_length/max(abs(dr_temp));
    r = r + dr*r_gain;
    r_max = max(reshape(abs(r), numel(r), 1));
    fprintf('max value in the perturbation is %.2f\n', r_max);
    
    % calculate the candidate for the next interation
    box_label = forward_propogation_det(x+r, rois, box_label, net); %generate qualified bbox for back-propagation (e.g., how many boxes are said it is a car)
    fprintf('remain %d boxes \n', sum(box_label(1,:) == box_label(3,:) & box_label(1,:) ~= 1));
    %   fprintf('remain %d boxes \n', sum(box_label(2,:) ~= box_label(3,:) & box_label(1,:) ~= 1));
    box_num(itr+1) = sum(box_label(1,:) == box_label(3,:) & box_label(1,:) ~= 1);
    
end

if (sum(box_label(1,:) == box_label(3,:) & box_label(1,:) ~= 1))
    status = 0;
else
    status = 1;
end

end