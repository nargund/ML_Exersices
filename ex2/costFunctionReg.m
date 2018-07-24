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

m = length(y); % number of training examples
%printf("No of training samples\n");
%m
%printf("Intial values of theta");
%theta
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%printf("Printing length of theta\n");
sztet=length(theta);%length of theta
thetat=theta';%transpose of theta
sigh=0;%sum of hmsq
for k=1:m,
	%printf("Values xi: ");
	x1=X(k,:);%returns K row values
	tx1=thetat*x1';%computing (theta)'*x accn to formulae [theta transpose x] 
	%printf("Computing Sigmoid of tx");
	h1=sigmoid(tx1);%calc hyposthesis of logistic regression
	yi1=y(k,:);%y(i) ie. correct answer
	cs1=((-yi1*log(h1))-((1-yi1)*(log(1-h1))));%computing cost at every row
	%printf("Val of hn: %f\n",hn);
	sigh=sigh+cs1;%summation of hmsq's
	end;
	
sumtet1=0;%sum of theta^2 values
cost = sigh/(m); %printing cost value lets hope for best
%printf("Removing theta(0) from matrix\n");
theta1=theta(2:end,1);%create another matrix by removing theta(0)
%printf("Size of theta1\n");
sztet1=length(theta1);%length of matrix with theta(0) removed
sumtet=0;%sumattion of theta^2
for k=1:sztet1,
	tetj=theta1(k,:);%to get theat(j) value
	tetj=tetj*tetj;%squaring it
	sumtet=sumtet+tetj;
end
regval=((lambda/(2*m))*(sumtet));%calc additonal delta regularization value
cost=cost+regval;%adding regularization value
J= cost;

for(tet=1:length(grad)),%traversing through all theta values
	grad1=0;%intial gardient value
	for k=1:m,
		%printf("Values xi: ");
		x2=X(k,:);%returns K row values
		tx2=thetat*x2';%computing (theta)'*x accn to formulae [theta transpose x] 
		%printf("Computing Sigmoid of tx");
		h2=sigmoid(tx2);%calc hyposthesis of logistic regression
		yi2=y(k,:);%y(i) ie. correct answer
		cs2=((h2-yi2)*x2(:,tet));%computing cost at every row 
		grad1=grad1+cs2;%appending gradients ie.partial diffrentiation
		%printf("Val of hn: %f\n",hn);
		%summation of hmsq's
	end;
	grad1=grad1/m;%NOTE: until this step common for all gardients
	if(tet==1)
		grad1=grad1+0;
	else
		grad1=grad1+((lambda/m)*(theta(tet,:)));
	endif
	grad(tet,:)=grad1;
end;


% =============================================================

end
