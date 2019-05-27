function [vecResult,bestlamada] = svmDML_choose_lamada(trainX,trainY,testX,testY,C1,C2,opt)
vecResult = [];
lamadabase = opt.lamadabase;
lamadarange = opt.lamadarange;
lamadalength = length(lamadarange);
opt.C1 = C1;
opt.C2 = C2;
for la = 1:1:lamadalength % find the best lamada
    lamada = lamadabase^lamadarange(la);
    opt.lamada = lamada;
    svmDML = svmDML_GBCD(trainX,trainY,opt);
    Resultlamada = svmDML_test(svmDML.svm,testX,testY);
    vecResult = [vecResult,Resultlamada.accuracy];
    disp( strcat( 'svmDML test with C1:',num2str(C1),'  C2:',num2str(C2),'  lamada:',num2str( lamadabase ),'^',num2str( lamadarange(la) ) ,'  itr==',num2str( svmDML.itr ) ));
    disp(strcat('lamada accuracy:',num2str(Resultlamada.accuracy),'...'));
    disp(strcat('train time:',num2str(svmDML.trainTime),'...'));
end
id = find( vecResult == max(vecResult) );
bestlamada = lamadabase^lamadarange( id(1) );
%bestlamada = lamadabase^lamadarange( id( length(id) ) );

end