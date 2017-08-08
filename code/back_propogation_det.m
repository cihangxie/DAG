function res = back_propogation_det(y, rois, box_label, net)
% do back-propoagation on qualified bbox
% --------------------------------------

obj_idx = unique(box_label(1,:));
obj_idx(obj_idx==1) = [];
res = zeros(size(y));

for i = 1:numel(obj_idx)
    % for i = 2
    rois_idx = (obj_idx(i) == box_label(1,:)) &  (box_label(2,:) ~= box_label(3,:)) & ((box_label(3,:) ~= 1));
    if sum(rois_idx) == 0
        continue
    end
    fooling_target = unique(box_label(2, rois_idx));
    obj_target = obj_idx(i);
    if numel(fooling_target) > 1
        error('must get something wrong!!!!')
    end
    net.blobs('data').reshape([size(y),1])
    net.blobs('rois').reshape([size(rois(:, rois_idx))])
    net.reshape();
    input_blobs = cell(2, 1);
    input_blobs{1} = y;
    input_blobs{2} = rois(:, rois_idx);
    net.forward(input_blobs); %do forward pass
    
    % do back-propogation
    dzdy = zeros(net.blobs(net.blob_names{end}).shape, 'single');
    dzdy(fooling_target, :) = 1;
    res_fool = net.backward({dzdy}); %do backward pass
    
    dzdy([fooling_target, obj_target], :) = dzdy([obj_target, fooling_target], :);
    res_obj = net.backward({dzdy}); %do backward pass
    
    res = res + (res_fool{1} - res_obj{1});
    
end


end

