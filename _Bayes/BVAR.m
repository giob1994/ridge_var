function [A, Sig_U, Sig_A, B, Sig_B] = BVAR(Yt, p, lambda, epsilon)
% BVAR Estimate a Bayesian VAR(p) model following Banbura & al. (2010)
%

T = size(Yt, 2);
K = size(Yt, 1);

X = zeros(K*p, T-p);
for i = 1:p
    X((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
end
% Regression matrix definition, [Banbura2010] eq.(3)
X = [X; ones(1, T-p)]'; % add intercept
Y = Yt(:,(p+1):T)';

% Estimate error variance
sigma = diag(Y' * (eye(T-p) - (X / (X' * X)) * X') * Y / T); 

% Dummy augmentation, [Banbura2010] eq.(5)
Yd = [ diag(sigma)/lambda; zeros(K*(p-1), K); diag(sigma); zeros(1, K) ];
Xd = [ kron(diag(1:p), diag(sigma)/lambda), zeros(K*p, 1);
       zeros(K, K*p+1);
       zeros(1, K*p), epsilon ];
Y = [Y; Yd];
X = [X; Xd];

% Posterior parameter mean, [Banbura2010] eq.(7)
B = (X' * X) \ X' * Y;
A = B(1:end-1,:)';       % w/out intercept coeffs

% Posterior parameter variance and residuals, [Banbura2010], eq.(7)
U = Y - X*B;
Sig_U = U'*U / (T-p-K*p-1);
Sig_B = kron(U'*U, inv(X'*X));          % Estimated varcov. of B
Sig_A = Sig_B(K+1:end, K+1:end);        % Estimated varcov. of A

end