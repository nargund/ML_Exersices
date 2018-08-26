function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
c_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sig_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]' ;

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
n=size(c_vec,1);
c_opt=0,sig_opt=0,min_val=99;
printf('Training model\n')
for i=1:n;
	for j=1:n;
		C=c_vec(i,1);
		sigma=sig_vec(j,1);
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		predictions= svmPredict(model, Xval);
		err=mean(double(predictions~=yval))
		if(err<min_val)
			min_val=err
			c_opt=C
			sig_opt=sigma
		end
	end
end
C=c_opt
sigma=sig_opt







% =========================================================================

end
