%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function svm = svmDML_svm(X,Y,dml,C1)
svm = struct();
LXr = X*dml.L';
%livsvm
libsvm_opt = ['-q -t 0 -c ',num2str(C1)];
model = svmtrain(Y, LXr, libsvm_opt);
wL = model.SVs'*model.sv_coef;
svm.wM = dml.L'* wL;
svm.model = model;
svm.b = -svm.model.rho;
end