function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%printf("Values of X matrix\n");
%X
%printf("Size of X matrix\n");
szx=size(X);%computing size of X matrix
%printf("Values of theta vector\n");
%theta
%printf("Size of theta matrix\n");
szt=size(theta);%size of theta
tetx=((X)*theta);%impleting theta' * X(theta'*X=X*theta)
hyp=sigmoid(tetx);%implementing hyptothesis for logistic regression
lghyp1=log(hyp);%performing ln on each element
lghyp2=log(1-hyp);%performing ln(1-hyp) on each element
%printf("Print elements of Y matrix\n");
%y
hfcst1=((-y).*(lghyp1));%computing 1st half of cost function ie. -y*log(h(x)) 
%because multiplication through each element ie. element wise multiplication
hfcst2=(-(1-y).*(lghyp2));%computing 2nd half of cost function ie. (1-y)*log(1-h(x))
semicst=hfcst1+hfcst2;%semi cost ie. before summation of each element
sumcst=sum(semicst(:));%summation of all 
cost=sumcst/m;
%printf("Values of theta 1\n");
theta1=theta(2:end,:);%create a new matrix by removing theta(0)
theta1=(theta1).*(theta1);%squaring each element in theta1
sumtet=sum(theta1(:));%summation of theta^2
regval1=((lambda)/(2*m));%regularization factor for cost function
sumtet=regval1*sumtet;%multiplying by reg factor
cost=cost+sumtet;
J=cost;
%to compute gradient
%printf("Print values of hypothesis\n");
%hyp
%printf("Values of Y\n");
%y
semigrad=hyp-y;%calc h(x)-y
%printf("Values of X\n");
%X
sumgrad=X'*semigrad;%calc summation semigrad and indudual "x" reqd for gradients
sumgrad=sumgrad./m;

sumgrad1=sumgrad(2:end,:);%gradients eleminating first element
theta2=theta(2:end);% new vector by removing theta(0)
regval2=(lambda/m);%computing regularization value for gradient
theta2=regval2*theta2;%multipyling by regularization value
sumgrad1=sumgrad1+theta2;%adding each element with regularization value
sumgrad(2:end,:)=sumgrad1(:);%updating values
grad(:)=sumgrad(:)

% =============================================================

grad = grad(:);

end
