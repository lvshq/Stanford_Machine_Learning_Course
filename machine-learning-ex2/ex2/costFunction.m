function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
one_vector = ones(m, 1);
h_theta = sigmoid(X * theta);
J = sum(-y .* log(h_theta) - (one_vector - y) .* log(1 - h_theta)) / m;
% The gradient is a vector, with each element being the gradient of certain
% theta(j). So here there should be a sum() function. Also be aware of the
% dimision of X and (h_theta - y), which we should use inverse multiply.
grad = X' * (h_theta - y) / m;

% =========================================================== ==

end
