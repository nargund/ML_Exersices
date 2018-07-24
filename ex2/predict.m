function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
thetat=theta';%calculating theta transpose
sz=length(X)%print the length value of x
%printing values of p
length(p)%printing length of p
for k=1:sz,
	x1=X(k,:);%returns a row column matrix conatning features of X
	h=thetat*(x1');%calc hyposthesis 
	sigval=sigmoid(h);
	if(sigval>0.5)
		p(k,:)=1;
	else
		p(k,:)=0;
	endif	
end;
printf("Printing values of P after updation\n");





% =========================================================================


end
