function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1, :)) % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z = sigmoid(X * theta)

% Fuck for no regularization on theta(0) !!
% J = -1 / m * (log(z)' * y + log(1 - z)' * (1 - y)) + lambda / (2 * m) * theta' * theta

% regularization without theta(0)
theta_reg = theta
theta_reg(1) = 0
J = -1 / m * (log(z)' * y + log(1 - z)' * (1 - y)) + lambda / (2 * m) * theta_reg' * theta_reg 

% grad(1) = 1 / m * (z - y)' * X(:, 1)
% for i = 2:n
%  grad(i) = 1 / m * (z - y)' * X(:, i) + lambda / m * theta_reg(i)
% end

grad = 1 / m * X' * (z - y) + lambda / m * theta_reg


% =============================================================

end
