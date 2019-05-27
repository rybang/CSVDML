%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = kernel_svmDML(X,C,opt)
switch opt.kernelType
case 'linear' 
    K = X*C;
case 'rbf' 

    [n,d] = size(X);
    [d,m] = size(C);
    K = zeros(n,m);
    for i=1:1:n
        for j=1:1:m
            K(i,j) = (X(i,:) - C(:,j)')*(X(i,:) - C(:,j)')';
        end
    end
    K = exp(-K./(2*opt.delta*opt.delta));
case 'rbf_fast' 

    XX = sum(X.*X,2);
    CC = sum(C'.*C',2);
    XC = X*C;
    K = abs(repmat(XX,[1 size(CC,1)]) + repmat(CC',[size(XX,1) 1]) - 2*XC);
    K = exp(-K./(2*opt.delta*opt.delta));
end
