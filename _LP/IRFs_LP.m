function [IRF,Std,sIRF,sStd] = IRFs_LP(Yt, q, H_irf)
%IRFs_LP Impulse response function estimation by Local Projection
%   Detailed explanation goes here

T = size(Yt, 2);
K = size(Yt, 1);

% Estimate VAR as preliminary step
[~, Sig_U_var] = LS_VAR(Yt, q);

P = chol(Sig_U_var)';

%% LPs Computation
% Follows from Kilian & Kim (2011), p 1460-1461

IRF  = zeros(K^2, H_irf+1);    % Impulse response functions
Std  = zeros(K^2, H_irf+1);    % Stdev. of IRFs
sIRF = zeros(K^2, H_irf+1);    % Structural IRFs
sStd = zeros(K^2, H_irf+1);    % Stdev. of structural IRFs

mL = GenEliminationMatrix(K);                % Elimination matrix
mD = GenDuplicationMatrix(K);                % Duplication matrix
mK = GenCommutationMatrix(K);                % Commutation matrix
H = mL'/(mL*(eye(K^2)+mK)*kron(P,eye(K))*mL');
Dplus = (mD'*mD)\mD';

Sigsig = 2*Dplus*kron(Sig_U_var,Sig_U_var)*Dplus';

vec = @(x) x(:);                             % Vec operation

IRF(:,1)  = vec(eye(K));
sIRF(:,1) = vec(P);
Std(:,1)  = 0;
sStd(:,1) = sqrt(diag(H*Sigsig*H'/(T-q)));

for h = 1:H_irf
    X = Yt(:,q:T-h);                    % Regressors of interest
    Z = zeros(K*(q-1)+1, T-q-h+1);      % Other lags
    Z(1,:) = ones(1,T-q-h+1);
    for i = 2:q
        Z(1+(1+(i-2)*K:(i-1)*K),:) = Yt(:,(q+1-i):(T-i-h+1));
    end
    Y = Yt(:,q+h:T);
    
    Mx = eye(T-q-h+1) - Z'/(Z*Z')*Z;    % Annihilation matrix
    % F1 = (X*Mx*X') \ (X*Mx*Y');         % LP estimator
    
    F = Y*[X;Z]' / ([X;Z]*[X;Z]');      % LP estimator (w/ all coeffs)
    
    IRF(:,h+1)  = vec(F(:,1:K));        % Reduced-form IRF
    sIRF(:,h+1) = vec(F(:,1:K)*P);      % Structural IRF
    
    % Newey-West estimator for variance (truncated at h)
    U = Y - F*[X;Z];                    % Residuals
    % U = U - mean(U,2);                  % Demean
    Sig_U_nw = U*U'/(T-q-K*q-1);        % R(0)
    
    m = h;
    for j = 1:m
        Rj = U(:,1:end-j)*U(:,j+1:end)'/(T-q);   % R(j)
        Sig_U_nw = Sig_U_nw + (1-j/(m+1))*(Rj+Rj'); 
    end    
    
    SigIRF = kron(P',eye(K))*kron(inv(X*Mx*X'),Sig_U_nw)*kron(P,eye(K));
    % NOTE: Kilian & Kim (2011) do as below since their data is transposed
    %   See formula (11), p. 1461:
    % SigIRF = kron(eye(K),P')*kron(Sig_U_nw,inv(X*Mx*X'))*kron(eye(K),P);
    
    % Second term, see formula (11), p. 1461
    GSigsigG = kron(eye(K),F(:,1:K))*H*Sigsig*H'*kron(eye(K),F(:,1:K)');
    
    Std(:,h+1) = sqrt(diag(SigIRF));
    sStd(:,h+1) = sqrt(diag(SigIRF+GSigsigG/(T-q)));
end
end