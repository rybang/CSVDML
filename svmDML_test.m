%Xt n*d matrix
%Yt n*1 vector
function result = svmDML_test(svm, X, Y)
%算出变换好的测试集X
Xt = X;
Yt = Y;

result.score = Xt*svm.wM + svm.b;
Y = sign(result.score);
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);  % 预测精度
