%Xt n*d matrix
%Yt n*1 vector
function result = svmDML_test(svm, X, Y)
%����任�õĲ��Լ�X
Xt = X;
Yt = Y;

result.score = Xt*svm.wM + svm.b;
Y = sign(result.score);
result.Y = Y;
result.accuracy = size(find(Y==Yt))/size(Yt);  % Ԥ�⾫��
