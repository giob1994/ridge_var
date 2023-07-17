function [A, Sig_U, Sig_A, U, B, Sig_B] = LS_VAR(Yt, p)
% LS_VAR OLS estimation of VAR(p) with data Yt  
% 

T = size(Yt, 2);
K = size(Yt, 1);

Z = zeros(K*p, T-p);
for i = 1:p
    Z((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
end
Z = [ones(1, T-p); Z]; % add intercept
Y = Yt(:,(p+1):T);

B = Y * Z' / (Z * Z'); % full coeffs
A = B(:, 2:end);       % w/out intercept coeffs

% Covariance estimation
U = Y - B * Z;                          % Residual matrix
Sig_U = U * U' / (T-p-K*p-1);           % Estimated variance
Sig_B = kron(inv(Z*Z'/(T-p)), Sig_U);   % Estimated varcov. of vec(B)
Sig_A = Sig_B(K+1:end, K+1:end);        % Estimated varcov. of vec(A)
end

