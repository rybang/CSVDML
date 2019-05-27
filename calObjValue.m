%objective value of the dual SVM problem
function objValue = calObjValue(X,Y,XC,XNNs,XNNd,svm,dml,lamada,C1,C2,inv_eps,opt)
wM = svm.wM;
b = svm.b;
M = dml.M;
r = dml.r;
invM = inverseM(M,inv_eps);
instanceNum = size(X,1);
tmp_GM = wM' * invM * wM;

objValue = 0.5 * tmp_GM + 0.5 * vec(M)' * vec(M);
if opt.isKernel ==1
    if opt.isDimReduced == 1
        JKJ = opt.J*opt.KtrainX*opt.J';
        objValue = 0.5 * tmp_GM + 0.5 * trace((JKJ*M)^2);
    else
        objValue = 0.5 * tmp_GM + 0.5 * trace((X*M)^2);
    end
end
Rs = 0;
for i = 1:1:instanceNum
    dist_c = XC(i,:)*M*XC(i,:)';
    Rs = Rs + dist_c;
end
%Rs = Rs/instanceNum;
objValue = objValue + lamada*Rs;

sai = 0;
score = X*wM;
score = score + b;
for i=1:1:length(score)
    temp = Y(i)*score(i);
    if(temp < 1 )
        sai = sai + 1 - temp;
    end
end
objValue = objValue + C1*sai;

eta_s = 0;
for i = 1:1:instanceNum
    dist_s = XNNs(i,:)*M*XNNs(i,:)';
    if dist_s > r-1
        eta_s = eta_s + dist_s -r + 1;
    end
end

eta_d = 0;
for i=1:1:instanceNum
    dist_d = XNNd(i,:)*M*XNNd(i,:)';
    if dist_d < r+1
        eta_d = eta_d + r + 1 - dist_d;
    end
end
eta = eta_s+eta_d;
objValue = objValue + C2*eta;
end