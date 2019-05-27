function [XNNs,XNNd,NN] = SearchNN(X,Y,opt)
[sampleNum,d]  = size(X);
isKernel = opt.isKernel;
if opt.fastSearchNN == 0
for i = 1:1:sampleNum
    sp=find(Y==Y(i));
    dp=find(Y~=Y(i));
    NNs_i = 0;
    NNd_i = 0;
    % search the nearest neighbor with the same label
    tmpX = X;
    tmpX(i,:) = Inf;
    dist = Inf;
    for j=1:1:length(sp)
        tmp = ( X(i,:) - tmpX(sp(j),:) )*( X(i,:) - tmpX(sp(j),:) )';
        if tmp < dist
            dist = tmp;
            NNs_i = sp(j);
        end
    end
    
    % search the nearest neighbor with the different label
    dist = Inf;
    for j=1:1:length(dp)
        tmp = ( X(i,:) - X(dp(j),:) )*( X(i,:) - X(dp(j),:) )';
        if tmp < dist
            dist = tmp;
            NNd_i = dp(j);
        end
    end
    if isKernel == 0
        XNNs(i,:) = X(i,:)- X(NNs_i,:);
        XNNd(i,:) = X(i,:)- X(NNd_i,:);
    else
        %XNNs(i,:) = kernel_svmDML(X(i,:)- X(NNs_i,:),X',opt);
        %XNNd(i,:) = kernel_svmDML(X(i,:)- X(NNd_i,:),X',opt);
        %XNNs(i,:) = kernel_svmDML(X(i,:),X',opt)-kernel_svmDML(X(NNs_i,:),X',opt);
        %XNNd(i,:) = kernel_svmDML(X(i,:),X',opt)-kernel_svmDML(X(NNd_i,:),X',opt);
%         XNNs(i,:) = ( opt.KtrainX(i,:)-opt.KtrainX(NNs_i,:) )*opt.J';
%         XNNd(i,:) = ( opt.KtrainX(i,:)-opt.KtrainX(NNd_i,:) )*opt.J';
        XNNs(i,:) =  opt.KtrainX(i,:)-opt.KtrainX(NNs_i,:) ;
        XNNd(i,:) =  opt.KtrainX(i,:)-opt.KtrainX(NNd_i,:) ;
    end
    NN(i).Matrix_XNNs = XNNs(i,:)'*XNNs(i,:);
    NN(i).Matrix_XNNd = XNNd(i,:)'*XNNd(i,:);
    NN(i).NNs_i = NNs_i;
    NN(i).NNd_i = NNd_i;
end
end


if opt.fastSearchNN == 1
    M = opt.M0;

    for i = 1:1:sampleNum
        sp=find(Y==Y(i));

        tmpX = X;
        tmpX(i,:) = Inf;
        tmpX_sp = tmpX(sp,:);
        [~,tmpNNs_i] = findNN(tmpX_sp',X(i,:)',M);
        NNs_i = sp(tmpNNs_i);
        XNNs(i,:) = X(i,:)- X(NNs_i,:);

        NN(i).Matrix_XNNs = XNNs(i,:)'*XNNs(i,:);
        NN(i).NNs_i = NNs_i;
    end

    uniqueY = unique(Y);
    for j = 1:1:length(uniqueY)
        samelabel_X_pos = find(Y==uniqueY(j));
        samelabel_X = X(samelabel_X_pos,:);
        differentlabel_X_pos = find(Y~=uniqueY(j));
        differentlabel_X = X(differentlabel_X_pos,:);
        [~,tmpNNd_vector] = findNN(differentlabel_X',samelabel_X',M);
        for jj = 1:1:length(samelabel_X_pos)
            NNd_i = differentlabel_X_pos( tmpNNd_vector(jj) );
            XNNd(samelabel_X_pos(jj),:) = X(samelabel_X_pos(jj),:)- X(NNd_i,:);

            NN( samelabel_X_pos(jj) ).Matrix_XNNd = XNNd(samelabel_X_pos(jj),:)'*XNNd(samelabel_X_pos(jj),:);
            NN( samelabel_X_pos(jj) ).NNd_i = NNd_i;
        end
    end
end
end