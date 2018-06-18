function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% The bias unit should be added as the first column in the input and hidden layers,
% and in the hidden layer it should be added after using the sigmoid() function.
L1_input = [ones(m, 1) X];
% Don't forget the sigmoid function.
L1_output = sigmoid(L1_input * Theta1');
L2_input = [ones(m, 1) L1_output];
L2_output = sigmoid(L2_input * Theta2');

% The max() function can return two values:
%   The maximum value in each row. We don't care about that.
%   The row index where the maximum value was found. That is our predicted
%      classification - the row where the maximum value was found.
[maximum, p] = max(L2_output, [], 2);

% =========================================================================


end
