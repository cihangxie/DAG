function softmaxA = softmax_dim(A,dim)
% apply the softmax function to a matrxi in certain dimension
% -----------------------------------------------------------

if nargin < 2, error('The dimension along which to do the softmax must be provided.'); end

s = ones(1, ndims(A));
s(dim) = size(A, dim);

% First get the maximum of A.
maxA = max(A, [], dim);
expA = exp(A-repmat(maxA, s));
softmaxA = expA ./ repmat(sum(expA,dim), s);
