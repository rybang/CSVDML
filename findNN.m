function [NN,NN_i]=findNN(Xtrain,Xtest,M)
    testSampleNum=size(Xtest,2);
    trainSampleNum=size(Xtrain,2);
    MXtrain=(M*Xtrain).*(Xtrain);
    sumxx=sum(MXtrain,1);
    sumxy=Xtrain'*M*Xtest;
    MXtest=(M*Xtest).*(Xtest);
    sumyy=sum(MXtest,1);
    dist=repmat(sumxx',1,testSampleNum)-2*sumxy+repmat(sumyy,trainSampleNum,1);
    [~, minindex]=min(dist);
    NN = Xtrain(:,minindex)';
    NN_i = minindex;
end