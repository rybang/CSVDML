%%cluster membership indicator function
function J=CMIF(trainX,trainY,opt)
% k is the number of clusters;
% J is a k*n matrix, n is the number of training instances.

%the type of cluster membership indicator function
isDimReduced = opt.isDimReduced; % 0 is no dimreduced, 1 is fully supervised and 2 is kmeans;
k = opt.kDimension;
J = zeros([k, length(trainY)]);
uniqueY = unique(trainY);
length_uY = length(uniqueY);
if isDimReduced == 2 %2 is kmeans;
    startMatrix = trainX(1:k,:);
    [IDX, C] = kmeans(trainX, k,'Start',startMatrix);
    for i =1:1:k
        pos = find(IDX==i);
        J(i,pos) = 1/length(pos);
    end
elseif isDimReduced == 1 %1 is fully supervised;
    if k <= length_uY
        for i=1:1:k
            pos = find(trainY==uniqueY(i));
            J(i,pos) = 1/length(pos);
        end
    end
    if k > length_uY 
        tf = floor(k/length_uY);
        tc = ceil(k/length_uY); % split the instances of each class into tc parts
        k_rest = k-tf*length_uY; 
        p_tf = 0;
        while 1 % process the first tf*length_uY cluster
            if tf == p_tf
                break;
            end
            base_num = p_tf*length_uY;
            for i=1:1:length_uY
                pos = find(trainY==uniqueY(i));
                interval_pos = floor(length(pos)/tc);
                if interval_pos == 0
                    error('the setting of the number of dimension is too big!!');
                end
                start_p = p_tf*interval_pos + 1;
                end_p = (p_tf+1)*interval_pos;
                if k_rest == 0
                    if p_tf + 1 == tf
                        end_p = length(pos);
                    end
                end
                J(base_num+i,pos(start_p:end_p)) = 1/length(pos(start_p:end_p));
            end
            p_tf = p_tf + 1;
        end
        base_num = tf*length_uY;
        if k_rest~=0 % process the last k_rest clusters
            p_class = 1;
            for i=1:1:k_rest
                set_vector = [];
                if i<k_rest
                    tempf = round(length_uY/k_rest); %every each tempf classes is used to construct one cluster
                    for j=1:1:tempf
                        pos = find(trainY==uniqueY(p_class));
                        interval_pos = floor(length(pos)/tc);
                        start_p = p_tf*interval_pos + 1;
                        end_p = length(pos);
                        set_vector = [set_vector,pos(start_p:end_p)'];
                        p_class = p_class + 1;
                    end
                else %the last few classes is used to construct the last one cluster
                    for j=p_class:1:length_uY
                        pos = find(trainY==uniqueY(p_class));
                        interval_pos = floor(length(pos)/tc);
                        start_p = p_tf*interval_pos + 1;
                        end_p = length(pos);
                        set_vector = [set_vector,pos(start_p:end_p)'];
                        p_class = p_class + 1;
                    end
                end
                J(base_num+i,set_vector) = 1/length(set_vector);
            end
        end
    end
%     if k > length_uY 
%         tf = floor(k/length_uY);
%         k_rest = k-length_uY;
%         p_tf = 0;
%         while 1
%             if tf == p_tf
%                 break;
%             end
%             base_num = p_tf*length_uY;
%              for i=1:1:length_uY
%                 pos = find(trainY==uniqueY(i));
%                 J(base_num+i,pos) = 1/length(pos);
%              end
%              p_tf = p_tf + 1;
%         end
%         base_num = tf*length_uY;
%          for i=1:1:k_rest
%             J(base_num+i,:) = 1/length(trainY);
%          end
%     end
else
    J = null;
end
end