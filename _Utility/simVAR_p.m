function [Yt] = simVAR_p(Ap, Sigma, N)
%simVAR_p Simulate a VAR(p) time series with given DGP parametes
%   It is assumed that parameter matrix Ap does not contain an 
%   intercept (so that Ap is of size [K \times (K*p)]

InitN = 100;

K = size(Ap, 1);
p = size(Ap, 2)/K;

% Put Ap into companion form
Apc = [Ap; eye(K*(p-1)), zeros(K*(p-1), K)];

Y0c = zeros(K*p, 1);
Ytc = zeros(K*p, N);

% Pre-sample simulation
for i = 1:InitN
    Y0c = Apc * Y0c;
    Y0c(1:K) = Y0c(1:K) + mvnrnd(zeros(K,1), Sigma)';
end    
% Sample simulation
Ytc(:,1) = Y0c;
for i = 2:N
    Ytc(:,i) = Apc * Ytc(:,i-1);
    Ytc(1:K,i) = Ytc(1:K,i) + mvnrnd(zeros(K,1), Sigma)';
end

Yt = Ytc(1:K,:);

end

