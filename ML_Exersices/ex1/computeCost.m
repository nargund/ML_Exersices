function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost

%printf("Val of x matrix\n");
%X %print X martix
%printf("Val of Y matrix\n"); 
%y %print y matrix
%printf("Val of theta\n");
%theta %print theta matrix
%printf("Value of m\n");
%m%print size or m value
%thetat=theta';%transpose of theta
%printf("Val of theta trans\n");
%thetat %print thetat
%printf('val of hyposthes1: \n');
thetat=theta';%transpose of theta
sigh=0;%sum of hmsq
for k=1:m,
	x1=X(k,:);%returns 1,2 matrix
	h=thetat*(x1');%calc hyposthesis of 1,2 * 2,1 = 1
	hn=h-y(k,:);%hypo(i)-y(i)
	
	%printf("Val of hn: %f\n",hn);
	hmsq=hn*hn;%square of hn
	sigh=sigh+hmsq;%summation of hmsq's
	end;
	
cost = sigh/(2*m); %printing cost value lets hope for best
J= cost;






% =========================================================================

end
