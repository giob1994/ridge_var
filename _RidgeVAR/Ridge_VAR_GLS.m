function [A, Sig_U, Sig_A, B, Sig_B] = Ridge_VAR_GLS(Yt, p, L)
%Ridge_VAR Estimate a VAR(p) model with data Yt via Ridge 
%

T = size(Yt, 2);
K = size(Yt, 1);

Z = zeros(K*p, T-p);
for i = 1:p
    Z((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
end
Z = [ones(1, T-p); Z]; % add intercept
Y = Yt(:,(p+1):T);

% Estimate error variance
Su_inv = inv(Y * (eye(T-p) - (Z' / (Z * Z')) * Z) * Y' / T); 

Lm = sparse(diag(kron([0; kron(L(:), ones(K,1))], ones(K,1))));
B = (kron(Z * Z', Su_inv) + Lm) \ kron(Z, Su_inv) * Y(:);
B = reshape(B, K, K*p+1);
A = B(:, 2:end);       % w/out intercept coeffs

% Covariance estimation
U = Y - B * Z;                          % Residual matrix
Sig_U = U * U' / (T-p-K*p-1);           % Residual covariance matrix
Sig_B = kron(inv(Z*Z'/(T-p)), Sig_U);   % Estimated varcov. of vec(B)
Sig_A = Sig_B(K+1:end, K+1:end);        % Estimated varcov. of vec(A)
end

