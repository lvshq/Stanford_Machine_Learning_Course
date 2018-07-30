function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

candidate_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
value_num = size(candidate_values, 2);
% Here the x1 and x2 params should not be initialzed.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
predict_error = mean(double(predictions ~= yval));
min_error = predict_error;

for i = 1:value_num
    for j = 1:value_num
        new_C = candidate_values(1, i);
        new_sigma = candidate_values(1, j);
        model= svmTrain(X, y, new_C, @(x1, x2) gaussianKernel(x1, x2, new_sigma)); 
        predictions = svmPredict(model, Xval);
        predict_error = mean(double(predictions ~= yval));
        if predict_error < min_error
            min_error = predict_error;
            C = new_C;
            sigma = new_sigma;
        end
    end
end


% =========================================================================

end
