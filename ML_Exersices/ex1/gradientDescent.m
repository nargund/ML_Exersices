function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
function [temp1,temp2] = semicost(X,y,theta,alpha,m)
%printf("Values of theta 1 and theta 2 in function:\n ");
%theta(1)
%theta(2)
thetat=theta';
a=0;
b=0;
sigh=0;
sigh1=0;
for k=1:m,
	
		x1=X(k,:);%returns 1,2 matrix
		h=thetat*(x1');%calc hyposthesis of 1,2 * 2,1 = 1
		hn=h-y(k,:);%hypo(i)-y(i)
		x=X(k,2);%assign value of x1
		hn1=hn*x;%computing hn*x1(i);
		%printf("Val of hn: %f\n",hn);
		%hmsq=hn*hn;%square of hn
		sigh=sigh+hn;%summation of hn
		sigh1=sigh1+hn1;%summation of hn1
		end;
		cost1 = sigh/m;%computing  1/m(sigh)
		cost2 = sigh1/m;%computing 1/m(sigh1)
		%printf("Val of temp1 and temp2 in function\n");
		temp1=alpha*cost1;
		temp2=alpha*cost2;
		%temp1=theta(1)-th
		%temp2=theta(2)-th1
		a=temp1;
		b=temp2;
		%thet1=theta(1);
		%thet2=theta(2);
end


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	d=semicost(X,y,theta,alpha,m);
	%printf("%f %f\n");
	%val of theta 0
	%printf("Val of temp of temp1 and temp2\n");
	a;
	b;
	%printf("Val of theta 1 and theta 2:\n");
	%theta(1)
	%theta(2)
	%printf("After updation\n");
	theta(1)=theta(1)-a;
	theta(2)=theta(2)-b;






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
%printf("Theta 0 = %f\n",theta(1));
%printf("Expected: -3.6303\n");

end
