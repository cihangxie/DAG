function box_label = assign_target_det(y, rois, gt, mapping, net)
% assign the adversarial target for each bbox
% -------------------------------------------

net.blobs('data').reshape([size(y),1])
net.blobs('rois').reshape([size(rois)])
net.reshape();
input_blobs = cell(2, 1);
input_blobs{1} = y;
input_blobs{2} = rois;
out = net.forward(input_blobs); %do forward pass

out = softmax_dim(out{1}, 1);

assignment_temp = zeros(size(gt,1), size(rois,2));
overlap_value = zeros(size(gt,1), size(rois,2));

for i = 1:size(gt,1)
    overlap_value(i,:) = boxoverlap(gt(i,2:5), rois(2:end,:)');
    % selecting positive targets
    assignment_temp(i,:) = (overlap_value(i,:) > 0.1) & (out(gt(i,1), :) > 0.1); 
end

% need to merge these indexes to a single row
assignment_idx = sum(assignment_temp,1);
[~, assignment] = max(assignment_temp, [], 1);
idx = find(assignment_idx>1); % means multiple assignment

if ~isempty(idx)
    for i =1:numel(idx)
       [~, assignment(idx(i))] = max(overlap_value(:,idx(i)).*assignment_temp(:,idx(i)));
    end
end

% first is what is its original label, second is its transfer label
box_label = zeros(3, size(rois,2)); 

% first row the original label, second row show the target label, third row show current prediction label
box_label(1, :) = gt(assignment,1);
box_label(1, assignment_idx==0) = 1;
box_label(2, :) = mapping(box_label(1,:));
box_label(3, :) = box_label(1, :);


end

