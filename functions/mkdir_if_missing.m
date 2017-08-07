function mkdir_if_missing( directory )
%MKDIR_IF_MISSING: make 'directory' if it does not exists.
%   Detailed explanation goes here
if ~exist(directory, 'dir')
    mkdir(directory);
end
end % end of function

