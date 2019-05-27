%function resultDmlSVM = itrOptimization_v2(X,Y,Xv,Yv,Xt,Yt,dml_svm_opt)
function svmDML = svmDML_GBCD(trainX,trainY,opt)
lamada = opt.lamada;
C1 = opt.C1;
C2 = opt.C2;
maxStopItr = opt.maxStopItr;
setpsize = opt.setpsize;
psd_eps = opt.psd_eps;
con_eps = opt.con_eps; %convergence epsilon
inv_eps = opt.inv_eps; %matrix with a small positive value inv_eps*I
factor = opt.factor;
vecobj = [];
X = trainX;
Y = trainY;

M0 = opt.M0;
[psdM,psdL] = makepsd(M0,psd_eps);
dml.L=psdL;
dml.M=psdM;
dml.r = opt.r;
itr = opt.itrOptNum;
XNNs = opt.trainXNNs; % similar NN
XNNd = opt.trainXNNd; % dissimilar NN
NN = opt.NN;

instanceNum = size(X,1);
%the matrix of X minus the center of X
XC = zeros(size(trainX));
meanX = mean(trainX);
for i=1:1:instanceNum
    XC(i,:) = trainX(i,:) - meanX;
end

svm = svmDML_svm(X,Y,dml,C1);
t1=clock;
stopItr = 0;
for i=1:1:itr

    %fix wM to train M
    Mt = dml.M;
    rt = dml.r;

    GM_c = zeros(size(Mt));
    for ins_i=1:1:instanceNum
        GM_c = GM_c + XC(ins_i,:)'*XC(ins_i,:);
    end
    %GM_c = GM_c/instanceNum;
    GM_c = lamada*GM_c;
    
    GM_s = zeros(size(Mt));
    Gr_s = 0;
    for ins_i = 1:1:instanceNum
        if XNNs(ins_i,:)*Mt*XNNs(ins_i,:)' <= rt-1
            continue;
        end
        %GM_s = GM_s + XNNs(ins_i,:)'*XNNs(ins_i,:);
        GM_s = GM_s + NN(ins_i).Matrix_XNNs;
        Gr_s = Gr_s + 1;
    end
    GM_s = C2*GM_s;
    Gr_s = C2*Gr_s;

    GM_d = zeros(size(Mt));
    Gr_d = 0;
    for ins_i=1:1:instanceNum
        if XNNd(ins_i,:)*Mt*XNNd(ins_i,:)' >= rt+1
            continue;
        end
        %GM_d = GM_d + XNNd(ins_i,:)'*XNNd(ins_i,:);
        GM_d = GM_d + NN(ins_i).Matrix_XNNd;
        Gr_d = Gr_d + 1;
    end
    GM_d = C2*GM_d;
    Gr_d = C2*Gr_d;

    % the gradient of M and r
    wM = svm.wM;
    %A = wM*wM' + inv_eps*eye(size(wM,1));
    A = wM*wM';
    invM = inverseM(Mt,inv_eps);
    tmp_GM = invM*A*invM;
    GM = -0.5 * tmp_GM + Mt + GM_c + GM_s - GM_d;
    if opt.isKernel ==1
        if opt.isDimReduced == 1
            JKJ = opt.J*opt.KtrainX*opt.J';
            GM = -0.5 * tmp_GM + JKJ*Mt*JKJ + GM_c + GM_s - GM_d;
        else
            GM = -0.5 * tmp_GM + X*Mt*X + GM_c + GM_s - GM_d;
        end
    end
    Gr = -Gr_s + Gr_d;

    tempM = Mt - setpsize*GM;
    %tempM = tempM + inv_eps*eye(size(tempM));
    tempr = rt - setpsize*Gr;

    % the projection of M_t+1 and r_t+1
    [psdM,psdL] = makepsd(tempM,psd_eps);
    dml.L=psdL;
    dml.M=psdM;
    if tempr < 1;
        %dml.r = 1;
        dml.r = tempr;
    else
        dml.r = tempr;
    end
    
    %fix M to train wM
    svm = svmDML_svm(X,Y,dml,C1);
    
    objValue = calObjValue(X,Y,XC,XNNs,XNNd,svm,dml,lamada,C1,C2,inv_eps,opt);
    vecobj = [vecobj,objValue];
%     if(i==1)
%         setpsize = setpsize * factor_inc;
%     end
    if(i>1)
        if abs( vecobj(i)-vecobj(i-1) ) < con_eps 
            stopItr = stopItr + 1;
        else
            stopItr = 0;
        end
        if vecobj(i)-vecobj(i-1) > 0 %if the value of object is increased, we should break it;
            dml = old_dml;
            svm = old_svm;
            break;
        end
%         if vecobj(i)-vecobj(i-1) < 0
%             setpsize = setpsize * factor_inc;
%         else
%             setpsize = setpsize * factor_dec;
%             dml = old_dml;
%             svm = old_svm;
%             vecobj(i) = vecobj(i-1);
%         end
    end
    if maxStopItr == stopItr
        break;
    end
    old_dml = dml;
    old_svm = svm;
    setpsize = setpsize * factor;
end
svmDML.vecobj = vecobj; % not necessary, just for test
svmDML.stopItr = stopItr; % not necessary, just for test
svmDML.itr = i; % not necessary, just for test
t2=clock;
trainTime=etime(t2,t1);
%disp(strcat('total training time:',num2str(trainTime),'s'));
svmDML.svm = svm;
svmDML.dml = dml;
svmDML.trainTime = trainTime;
end
