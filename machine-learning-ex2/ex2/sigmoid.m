function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% @(x) is anonymous function, apply its following function to each element x in z
g = arrayfun(@(x) 1.0 / (1 + exp(-x)), z);

% =============================================================

end
