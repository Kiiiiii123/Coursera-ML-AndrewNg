function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
%原来的X和Theta已经被合并在一起并展开成了一个向量，这里需要先将其转变成能用的矩阵
%不同的电影拥有不同的特征数值
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
#不同的用户拥有不同的预判参数
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
J=sum(sum(((X*Theta'-Y) .^2) .*R))/2+lambda*(sum(sum(X .^2))+sum(sum(Theta .^2)))/2;
%非向量化写法
%从全部电影出发
%for i=1:size(X,1)
%  idx1=find(R(i,:)==1);
%  ThetaTemp=Theta(idx1,:);
%  YTemp1=Y(i,idx1);
%  X_grad(i,:)=(X(i,:)*ThetaTemp'-YTemp1)*ThetaTemp+lambda*X(i,:);
%endfor
%从全部用户出发
%for j=1:size(Theta,1)
%  idx2=find(R(:,j)==1);
%  XTemp=X(idx2,:);
%  YTemp2=Y(idx2,j)';
%  Theta_grad(j,:)=(Theta(j,:)*XTemp'-YTemp2)*XTemp+lambda*Theta(j,:);
%endfor
X_grad=((X*Theta'-Y) .*R)*Theta+lambda*X;
Theta_grad=((X*Theta'-Y) .*R)'*X+lambda*Theta;
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
