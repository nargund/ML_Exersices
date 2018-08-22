function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X];

% Now, implement forward propagation, similar to ex3
%function to return hypothesis
function [hyp]=hypothesis(Theta1,Theta2,xi)
	x=xi';%transpose of X so its 401x500;
	z2=Theta1*x;%we know x1...xn =a(1) ie. z2=theta1*a(1)
	Z2=z2;
	a2_int=sigmoid(z2);%ie a(2)=g(z2) computing intial a(2)
	%printf("Size of a(2)\n");
	size(a2_int);%returns 25x5000 matrix ie 25=no of rows in 2nd layer
	%5000 = input rows
	a2_int2=[ones(size(a2_int)(2), 1)';a2_int];%adding first row full of ones
	size(a2_int2);
	A2=a2_int2;
	%a2_final=a2_int2';%making the matrix 5000 * 26 (26 because multiplying with Theta2)
	%ie. adding bias value
	%printf("Size of a2_mid\n");
	%size(a2_final)
	%NOTE no of rows in any matrix a(1)..a(n) should be equal to len of x
	%no of columns should be equal to number of theta here a2_int=5000x25
	%a2_final = 5000x26
	z3=Theta2*a2_int2;%we know a(2)1...a(n)n =a(2) ie. z3=theta1*a(2)
	Z3=z3;
	a3_int=sigmoid(z3);%hypthesis wrt to layer3 ie. output layer
	%printf("Size of a3_int\n");
	size(a3_int); %returns 10x5000 matrix
	%[max_value,max_index]=max(a3_final,[],2);%getting index of max value same logic 
	%as one vs all refer predict onevsall in the same directory
	%p=max_index;
	hyp=a3_int;
end

%function to return vector according to output
function [Y]=Y(num)
	Y=zeros(num_labels,1);
	Y(num,:)=1;
end

%implementing cost function
sigcostm=0;
for i=1:m,
	hyp_i=hypothesis(Theta1,Theta2,X(i,:));%contains hypthesis of particular row
	y_i=Y(y(i,:));%contains vector of Y
	sigcostk=0;
	for k=1:num_labels,
		hypi_k=hyp_i(k,:);%returns output of kth element
		yi_k=y_i(k,:);%returns kth element
		costk1=(-yi_k)*(log(hypi_k));%applying formulae
		costk2=(1-yi_k)*(log(1-hypi_k));
		costk=costk1-costk2;
		sigcostk=sigcostk+costk;%summation over k terms wrt to o/p
	end
	sigcostm=sigcostm+sigcostk;%summation over m terms wrt to i/p
end
fin_cost=(sigcostm)/(m);
J=fin_cost;
size(Theta1);
tet1= Theta1(:,2:size(Theta1,2));%remove first column ie theta(n,0) when n=row number
%printf("Size of theta1");
size(tet1);
size(Theta2);
tet2= Theta2(:,2:size(Theta2,2));%remove first column ie theta(n,0) when n=row number
%printf("Size of theta2");
size(tet2);
L=3;%total number of layers
tet_sum=0;
for l=1:L-1,%determining which layer
	if(l==1)
		tet_l=tet1;%retriving particlular theta(l)
	else
		tet_l=tet2;
	endif
	for I=1:(size(tet_l,2)),%determininig i value
		%printf("I value =%d",i);
		for J=1:(size(tet_l,1)),%determining j value
			%printf("k value:%d",k);
			tet_sq=tet_l(J,I);%applying formulae
			tet_sq1=tet_sq*tet_sq;%squaring
			tet_sum=tet_sum+tet_sq1;
		end
	end
end
reg_val1=(lambda/(2*m));
reg_val=reg_val1*tet_sum;
fin_cost=fin_cost+reg_val;
J=fin_cost;


%implementing BackProp
DEL1=0;
DEL2=0;
for t=1:m,
	x=X(t,:)';
	a1=x;
	%x=xi';%transpose of X so its 401x500;
	z2=Theta1*x;%we know x1...xn =a(1) ie. z2=theta1*a(1)
	a2_int=sigmoid(z2);%ie a(2)=g(z2) computing intial a(2)
	a2=[ones(size(a2_int)(2), 1)';a2_int];%adding first row full of ones
	z3=Theta2*a2;
	a3=sigmoid(z3);
	yi_t=Y(y(t,:));
	del3=a3-yi_t;
	z2=[1;z2];%adding bias
	de2=(Theta2'*del3).*sigmoidGradient(z2);
	de2=de2(2:end);%del_n(0) terms
	DEL1=DEL1+(de2*a1');
	DEL2=DEL2+(del3*a2');
end
Theta1_grad=1/m*(DEL1);
Theta2_grad=1/m*(DEL2);
tet_sum=0;

for C=1:(size(tet1,2)),%determininig i value
	%printf("I value =%d",i);
	for R=1:(size(tet1,1)),%determining j value
		%printf("k value:%d",k);
		tet_sq3=tet1(R,C);%applying formulae
		reg_tet1(R,C)=tet_sq3;
	end
end
for C=1:(size(tet2,2)),%determininig i value
	%printf("I value =%d",i);
	for R=1:(size(tet2,1)),%determining j value
		%printf("k value:%d",k);
		tet_sq4=tet2(R,C);%applying formulae
		reg_tet2(R,C)=tet_sq4;%creating a colummn matrix of theta values
	end
end
p1=size(reg_tet1,1);
p2=size(reg_tet2,1);
reg_tet1=[zeros(p1, 1) reg_tet1];%make first column =0
reg_tet2=[zeros(p2, 1) reg_tet2];%make first column = 0
reg_tet1=(lambda/m).*reg_tet1;
reg_tet2=(lambda/m).*reg_tet2;
Theta1_grad=Theta1_grad+reg_tet1;
Theta2_grad=Theta2_grad+reg_tet2;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
