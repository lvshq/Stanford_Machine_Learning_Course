function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
[unreg_cost, unreg_grad] = costFunction(theta, X, y);
% Should not compute theta(1)
regularized_cost = lambda / (2*m) * sum(theta(2:end) .^ 2);
J = unreg_cost + regularized_cost;
%for index_theta = 1:size(theta, 1)
    %grad(index_theta) = (X' * (h_theta - y) + lambda * theta) / m;
%end

grad = unreg_grad + (lambda * theta) / m;
grad(1) = unreg_grad(1);

% =============================================================

end
