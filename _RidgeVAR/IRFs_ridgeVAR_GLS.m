function [IRF,Std,P,sIRF,sStd,sCov,A,Sig_A,U] = IRFs_ridgeVAR_GLS(Yt, p, L, H_irf)
%IRFs_ridgeVAR IRF estimation for VAR model via lag-adapt. ridge 
%   Detailed explanation goes here

T = size(Yt, 2);
K = size(Yt, 1);

Z = zeros(K*p, T-p);
for i = 1:p
    Z((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
end
Z = [ones(1, T-p); Z];
Y = Yt(:,(p+1):T);

% Estimate error variance
Su_inv = inv(Y * (eye(T-p) - (Z' / (Z * Z')) * Z) * Y' / T); 

Lm = sparse(diag(kron([0; kron(L(:), ones(K,1))], ones(K,1))));
B = (kron(Z * Z', Su_inv) + Lm) \ kron(Z, Su_inv) * Y(:);
B = reshape(B, K, K*p+1);
A = B(:, 2:end);       % w/out intercept coeffs

% Covariance estimation
U = Y - B * Z;                          % Residual matrix
Sig_U = U * U' / (T-p-K*p-1);           % Estimated variance
Sig_B = kron(inv(Z*Z'/(T-p)), Sig_U);   % Estimated varcov. of vec(B)
Sig_A = Sig_B(K+1:end, K+1:end);        % Estimated varcov. of vec(A)

% Estimate VAR residuals for structural decomposition
B_ls = Y * Z' / (Z * Z'); 
U_ls = Y - B_ls * Z;
Sig_U_ls = U_ls * U_ls' / (T-p-K*p-1);

%% IRFs computation
% Follows from Lutkepohl (2005), pp 108-113

IRF  = zeros(K^2, H_irf+1);    % Impulse response functions
Std  = zeros(K^2, H_irf+1);    % Stdev. of IRFs
sIRF = zeros(K^2, H_irf+1);    % Structural IRFs
sStd = zeros(K^2, H_irf+1);    % Stdev. of structural IRFs
sCov = zeros(K^2, H_irf+1);    % Cov. of structural IRFs due to Sig_U

Ac = [A; eye(K*(p-1)), zeros(K*(p-1), K)];   % Companion form

P = chol(Sig_U_ls)';                            % Structural assumption

J = [eye(K), zeros(K, K*(p-1))];
mL = GenEliminationMatrix(K);                % Elimination matrix
mD = GenDuplicationMatrix(K);                % Duplication matrix
mK = GenCommutationMatrix(K);                % Commutation matrix
H = mL'/(mL*(eye(K^2)+mK)*kron(P,eye(K))*mL');
Dplus = (mD'*mD)\mD';

vec = @(x) x(:);                             % Vec operation
% res = @(x) reshape(x,K,K);                   % Reshape-[K x K] operation 

Sigsig = 2*Dplus*kron(Sig_U,Sig_U)*Dplus';
Cbar0  = kron(eye(K),eye(K))*H;

IRF(:,1)  = vec(eye(K));
sIRF(:,1) = vec(P);
Std(:,1)  = 0;
sStd(:,1) = sqrt(diag(Cbar0*Sigsig*Cbar0')/(T-p));
sCov(:,1) = diag(Cbar0*Sigsig*Cbar0'/(T-p));

Ac_pow = zeros(K*p,K*p,H_irf+1);
Ac_pow(:,:,1) = eye(size(Ac));

for h = 1:H_irf
    Ac_pow(:,:,h+1) = Ac*Ac_pow(:,:,h);
    
    Gi = zeros(K^2,K^2*p);
    for m = 0:(h-1)
        % Gi = Gi + kron(J*(Ac')^(h-1-m), J*(Ac^m)*J');
        Gi = Gi + kron(Ac_pow(:,1:K,h-m)', Ac_pow(1:K,1:K,m+1));
    end
    Phi_i = Ac_pow(1:K,1:K,h+1);      % = J*(Ac^h)*J';
    Ci    = kron(P',eye(K))*Gi;
    Cbari = kron(eye(K),Phi_i)*H;
    
    IRF(:,h+1)  = vec(Phi_i);
    sIRF(:,h+1) = vec(Phi_i*P);
    Std(:,h+1)  = sqrt(diag(Gi*Sig_A*Gi')/(T-p));
    sStd(:,h+1) = sqrt(diag(Ci*Sig_A*Ci'+Cbari*Sigsig*Cbari')/(T-p));
    sCov(:,h+1) = diag(Cbari*Sigsig*Cbari')/(T-p);
end
end