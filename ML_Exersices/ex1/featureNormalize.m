function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m=length(X);
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%
%printf("Intial values of X\n");
%X
%printf("Mean of x1\n");       
mx1=mean(X(:,1));%calculating the mean of x1
mu(1,1)=mx1;%storing mx1 in mu(1,1)
%printf("Mean of x2\n");
mx2=mean(X(:,2));%calculating mean of x2
mu(1,2)=mx2;%storing mx2 in mu(1,2)

%printf("std dev of x1\n");
sd1=std(X(:,1));%calc standard deviation of x1
sigma(1,1)=sd1;%storing in sigma matrix of std dev
%printf("Std dev of x2\n");
sd2=std(X(:,2));%calc std dev of x2
sigma(1,2)=sd2;%storing in sigma matrix

for k=1:m,
	tx1=0;
	tx2=0;
	tx1=X(k,1)-mu(1,1);%subs with mean of x1
	X(k,1)=tx1/sigma(1,1);%divide by std dev and update of x1
	tx2=X(k,2)-mu(1,2);%subs with mean of x2
	X(k,2)=tx2/sigma(1,2);%divivde by std dev of x2 and update
	end
	
printf("Updated vals of X\n");
X








% ============================================================

end
