function [J, grad] = lrCostFunction(phi, X, y)

    %-- LRCOSTFUNCTION Compute cost and gradient for logistic regression
    %--   J = LRCOSTFUNCTION(phi, X, y, lambda) computes the cost of using
    %--   theta as the parameter for logistic regression and the
    %--   gradient of the cost w.r.t. to the parameters. 

    
    %-- Initialization of energy value J
    J = 0;

    %-- Initialization of gradient vector
    [m,n] = size(X);
    grad = zeros(1,n);
    
    
    % ====================== YOUR CODE HERE =========================
    % YOU SHOULD COMPUTE 
    %   - THE VALUE OF THE ENERGY FUNCTION J
    %   - THE GRADIENT VECTOR OF THE ENERGY FUNCTION J
    % ===============================================================
    y = (y==1);
    h = lrc.sigmoid(X*phi');
    J = -(1/m)*(y'*log(h)+(1-y)'*log(1-h));
    %disp(J)
    
    h = lrc.sigmoid(X*phi');
    grad2 = (1/m)*(X'*(h-y));
    grad = grad2' + grad;
    %disp(grad*grad')


end



