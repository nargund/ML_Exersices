function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hyp_mat=X*theta;%implementing theta'*X
cost_mat=hyp_mat-y;%hyp -y(i)
cost_matsq=cost_mat.*cost_mat;%squared errors
sig_cost=sum(cost_matsq);%summation of terms
cost_1=(1/(2*m))*sig_cost;%computing intial cost w/o regularization
tet=theta(2:end);%removing theta(0) from matrix
tet=tet.*tet;
sig_tet=sum(tet);%computing sugma value of theta 
reg_val=(lambda/(2*m));%computing regularization term
cost_2=reg_val*sig_tet;
J=cost_1+cost_2;

hyp_mat_grad=X'*cost_mat;%implemrnting sigma((hyp(i)-y(i))*xj)
%note X is 12x1 and hyp_mat is 12x1
hyp_mat_grad=hyp_mat_grad./m;%dividing each elem by m
reg_val_grad=(lambda/m);
tet2=theta(2:end);
tet2=reg_val_grad*tet2;%computing (lambda/m)*theta(j)
hyp_mat_grad1=hyp_mat_grad(2:end,:);
hyp_mat_grad1=hyp_mat_grad1+tet2;
hyp_mat_grad(2:end,:)=hyp_mat_grad1(:);
grad(:)=hyp_mat_grad(:);
%sig_grad=sum(hyp_mat_grad);%computing sigma of hyp_mat_grad


















% =========================================================================

grad = grad(:);

end
