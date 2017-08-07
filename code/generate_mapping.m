function [mapping, target_idx_candidate_shuffle] = generate_mapping(gt_idx)

% generating random mapping
target_idx_candidate = setdiff(1:20, gt_idx);
target_idx_candidate_shuffle = target_idx_candidate(randperm(length(target_idx_candidate)));
mapping = zeros(1, 20);
mapping(gt_idx) = target_idx_candidate_shuffle(1:length(gt_idx));

end