function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
sz=size(z);%size of the matrix z
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%to compute basic sigmoid fn
%z%displaying z matrix
%g%displaying g matrix

k1=sz(:,1);%k1 = number of rows
k2=sz(:,2);%k2 = number of columns 
for r=1:k1,
	for c=1:k2, %ie. Traversing through every element
		a=z(r,c);
		epw=exp(-a);%storing e^-z value
		f=(1/(1+epw));%computing sigmoid function
		g(r,c)=f;
	end;
end;
%printf("The values of sigmoid function of every value in matrix: ");

% =============================================================

end
