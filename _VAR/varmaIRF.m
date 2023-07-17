function [IRF,sIRF] = varmaIRF(A,M,P,H_irf)
%varmaIRF Compute IRFs and structural IRFs from VARMA model specification
%   Detailed explanation goes here

K = size(A, 1);     % Dimensions
p = size(A, 2)/K;   % VAR lags
q = size(M, 2)/K;   % VMA lags

% VARMA companion form
A11 = [A; eye(K*(p-1)), zeros(K*(p-1), K)];  
A12 = [M; zeros(K*(p-1), K*q)];
A21 = zeros(K*q, K*p);
A22 = [zeros(K,K*q); eye(K*(q-1)), zeros(K*(q-1), K)];
A = [A11, A12; A21, A22];

J = [eye(K), zeros(K, K*(p+q-1))];
H = [eye(K); zeros(K*(p-1),K); eye(K); zeros(K*(q-1),K)];

%% IRFs and sIRFs 

IRF  = zeros(K^2, H_irf+1);
sIRF = zeros(K^2, H_irf+1);

IRF(:,1)  = reshape(eye(K),K^2,1);
sIRF(:,1) = reshape(eye(K)*P,K^2,1);
for h = 1:H_irf
    Phi_i = J*(A^h)*H;
    IRF(:,h+1) = reshape(Phi_i,K^2,1);
    sIRF(:,h+1) = reshape(Phi_i*P,K^2,1);
end
end

