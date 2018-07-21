function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part 1: CostFunction
% -------------------------------------------------------------
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% Note: Here the cost is the difference between the probability and the corresbonding 1s or 0s,
%    instead of the diff between the class number (1-10). So bellow code in ex3 should not be used.
% [maximum, output] = max(a3, [], 2);

% eye_matrix is 10*10 matrix.
eye_matrix = eye(num_labels);
% Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5),
%    namely mapping this vector with values 1...K into a binary vector of
%    1's and 0's .
% y_matrix is 5000*10, while y is 5000*1
y_matrix = eye_matrix(y, :);
% sum(X,2) means sum up each columns.
unreg_cost = sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3), 2)) / m;
% Note that Theta1/Theta2 are matrics, so we should use sum() twice:
%   first sum add all rows, second sum add all elements in the row.
regularized_cost = lambda / (2*m) * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)));
J = unreg_cost + regularized_cost;


% Part 2: Backpropagation
% -------------------------------------------------------------

delta3 = a3 - y_matrix;
% Since the bias unit cannot propagate error back, so we should eliminate
%   the first column.
cutted_Theta2 = Theta2(:,2:end);
delta2 = (delta3*cutted_Theta2) .* sigmoidGradient(z2);

Delta1 = delta2' * a1;
Delta2 = delta3' * a2;

% We should not be regularizing the first column of жи(l) which
%    is used for the bias term
cutted_Theta1 = Theta1;
cutted_Theta1(:, 1) = cutted_Theta1(:, 1) * 0;
cutted_Theta2 = Theta2;
cutted_Theta2(:, 1) = cutted_Theta2(:, 1) * 0;

Theta1_grad = Delta1 / m + lambda / m * cutted_Theta1;
Theta2_grad = Delta2 / m + lambda / m * cutted_Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
