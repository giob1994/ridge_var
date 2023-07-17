function [Yt] = simVARMA_p_1(Ap, M, Sigma, N)
% simVARMAp1 Simulate a VARMA(p,1) with parameter matrices Ap, M
%   It is assumed that parameter matrix Ap does not contain an 
%   intercept (so that Ap is of size [K \times (K*p)]. Using M, 
%   define -M to be the MA(1) matrix. 
%                                      (see Kilian & Lutkepohl, 2017, p.45)

InitN = 5000;

K = size(Ap, 1);
p = size(Ap, 2)/K;

% Put Ap into companion form
Apc = [Ap; eye(K*(p-1)), zeros(K*(p-1), K)];

Y0c = zeros(K*p, 1);
Ytc = zeros(K*p, N);

% Ulag0 = zeros(K, 1);
Ulag1 = mvnrnd(zeros(K,1), Sigma)'; 

% Pre-sample simulation
for i = 1:InitN
    Y0c = Apc * Y0c;
    Ulag0 = mvnrnd(zeros(K,1), Sigma)';
    Y0c(1:K) = Y0c(1:K) + Ulag0 + M * Ulag1;
    Ulag1 = Ulag0;
end    
% Sample simulation
Ytc(:,1) = Y0c;
for i = 2:N
    Ytc(:,i) = Apc * Ytc(:,i-1);
    Ulag0 = mvnrnd(zeros(K,1), Sigma)';
    Ytc(1:K,i) = Ytc(1:K,i) + Ulag0 + M * Ulag1;
    Ulag1 = Ulag0;
end

Yt = Ytc(1:K,:);

end

