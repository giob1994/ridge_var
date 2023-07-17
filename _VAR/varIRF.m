function [IRF,sIRF] = varIRF(A,P,H_irf)
%varmaIRF Compute IRFs and structural IRFs from VARMA model specification
%   Detailed explanation goes here

K = size(A, 1);     % Dimensions
p = size(A, 2)/K;   % VAR lags

% VARMA companion form
A = [A; eye(K*(p-1)), zeros(K*(p-1), K)];  

J = [eye(K), zeros(K, K*(p-1))];

%% IRFs and sIRFs 

IRF  = zeros(K^2, H_irf+1);
sIRF = zeros(K^2, H_irf+1);

IRF(:,1)  = reshape(eye(K),K^2,1);
sIRF(:,1) = reshape(eye(K)*P,K^2,1);
A_pow = eye(K*p);
for h = 1:H_irf
    A_pow = A * A_pow;
    Phi_i = J*A_pow*J';
    IRF(:,h+1) = reshape(Phi_i,K^2,1);
    sIRF(:,h+1) = reshape(Phi_i*P,K^2,1);
end
end

