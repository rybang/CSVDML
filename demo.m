clear;
close all;

load breast-test.mat

fprintf('\ntraining the CSV-DML classifier.\n');

C1base = 2;
C1range = -5:1:5;

C2base = 10;
C2range = -5:1:5;

lamadabase = 10;
lamadarange = -5:1:5;
opt.setpsize = 1e-1; % selected from {1e-0,1e-1,1e-2,1e-3,1e-4,1e-5}
opt.M0_type = 1; %1 eu distance and 2 is ma distnace

opt.isDimReduced = 1; % 0 is no dimReduced, 1 is fully supervised and 2 is kmeans;
opt.kDimension = 20; % the number of dimension
opt.isKernel = 1; % 0 is non-kernel, 1 is kernel
%opt.kernelType = 'linear';
opt.kernelType = 'rbf_fast';
opt.delta = 1e-0;

initC2 = C2base^0;
initlamada = lamadabase^0;
opt.C1base = C1base;
opt.C1range = C1range;
opt.C2base = C2base;
opt.C2range = C2range;
opt.lamadabase = lamadabase;
opt.lamadarange = lamadarange;

opt.itrOptNum = 15;%
opt.maxStopItr = 3;
opt.psd_eps = 1e-10;
opt.con_eps = 1e-1; %convergence epsilon
opt.inv_eps = 1e-8; %matrix with a small positive value inv_eps*I
opt.factor = 0.9*1.01;

trainX_original = trainX;

if opt.isKernel ==1
    trainX = kernel_svmDML(trainX,trainX_original',opt);
    testX = kernel_svmDML(testX,trainX_original',opt);
    opt.KtrainX = trainX;
    if opt.isDimReduced == 1
        %uY = unique(trainY);
        %opt.kDimension = uY;
        opt.J = CMIF(trainX_original,trainY,opt);
        trainX = trainX*opt.J';
        testX = testX*opt.J';
    end
end

% nearst neighbor search
tempM0 = eye(size(trainX,2));
tempr = 1;
opt.M0 = tempM0;
opt.r = tempr;
opt.fastSearchNN = 1;
t1=clock;
[trainXNNs,trainXNNd,NN] = SearchNN(trainX,trainY,opt);
%[trainXNNs,NN] = SearchNN(trainX_original,trainY,opt);
opt.trainXNNs = trainXNNs; % similar NN
opt.trainXNNd = trainXNNd; % dissimilar NN
opt.NN = NN;
t2=clock;
NNTime2=etime(t2,t1);

% find the best C1
[vecResult2,bestC1] = svmDML_choose_C1(trainX,trainY,testX,testY,initC2,initlamada,opt);
fprintf('.\n')
% find the best lamada
[vecResult3,bestlamada] = svmDML_choose_lamada(trainX,trainY,testX,testY,bestC1,initC2,opt);
fprintf('...\n')
% find the best C2
[vecResult1,bestC2] = svmDML_choose_C2(trainX,trainY,testX,testY,bestC1,bestlamada,opt);
fprintf('..\n')

%final train with the best parameters
opt.C1 = bestC1;
opt.C2 = bestC2;
opt.lamada = bestlamada; 
t1=clock;
svmDML = svmDML_GBCD(trainX,trainY,opt);
t2=clock;
trainTime=etime(t2,t1);

finalResult = svmDML_test(svmDML.svm,testX,testY);

disp(strcat( 'CSV-DML test with bestlamada:',num2str( bestlamada ),'  bestC1:',num2str(bestC1),'  bestC2:',num2str(bestC2) ));
disp(strcat('final accuracy:',num2str(finalResult.accuracy),'...'));

disp(strcat('final training time:',num2str(trainTime),'...'));
%keyboard;
