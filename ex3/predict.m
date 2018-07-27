function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
m
num_labels = size(Theta2, 1);

X=[ones(m,1) X];

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
printf("Size of Theta 1\n");
size(Theta1)
printf("Size of Theta 2\n");
size(Theta2)
%printf("Values of Theta 2\n");
Theta2;
printf("Size of X matrix\n");
size(X)
x=X';%transpose of X so its 401x500
z2=Theta1*x;%we know x1...xn =a(1) ie. z2=theta1*a(1)
a2_int=sigmoid(z2);%ie a(2)=g(z2) computing intial a(2)
printf("Size of a(2)\n");
size(a2_int)%returns 25x5000 matrix ie 25=no of rows in 2nd layer
%5000 = input rows
a2_int2=[ones(size(a2_int)(2), 1)';a2_int];%adding first row full of ones
size(a2_int2)
a2_final=a2_int2';%making the matrix 5000 * 26 (26 because multiplying with Theta2)
%ie. adding bias value
printf("Size of a2_mid\n");
size(a2_final)
%NOTE no of rows in any matrix a(1)..a(n) should be equal to len of x
%no of columns should be equal to number of theta here a2_int=5000x25
%a2_final = 5000x26
z3=Theta2*a2_final';%we know a(2)1...a(n)n =a(2) ie. z3=theta1*a(2)
a3_int=sigmoid(z3);%hypthesis wrt to layer3 ie. output layer
printf("Size of a3_int");
size(a3_int) %returns 10x5000 matrix
a3_final=a3_int';
[max_value,max_index]=max(a3_final,[],2);%getting index of max value same logic 
%as one vs all refer predict onevsall in the same directory
p=max_index;



% =========================================================================


end
