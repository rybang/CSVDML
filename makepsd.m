%copy from mLMNN2.5.1-WithMT压缩包里面的mtlmnn.m文件
%function [M,L]=makepsd(Q,pars)
function [M,L]=makepsd(Q,epsilon)
    Q = (Q+Q')/2;
    % decompose Q
    [L,dd]=eig(Q);
    dd=real(diag(dd));
    L=real(L);
    % reassemble Q (ignore negative eigenvalues)
    j=find(dd<epsilon);
%     if(~isempty(j)) 
%         if(~pars.quiet)fprintf('[%i]',length(j));end;
%     end;
    dd(j)=0;
    [temp,ii]=sort(-dd);
    L=L(:,ii);
    dd=dd(ii);
    
    L=(L*diag(sqrt(dd)))';
    M=L'*L;
end