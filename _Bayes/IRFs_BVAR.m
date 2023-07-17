function [IRF,IRFdraws,P,sIRF,sIRFdraws,B,Sig_B,U] = ...
                            IRFs_BVAR(Yt, p, lambda, epsilon, H_irf)
%IRFs_BVAR Impulse response function estimation for Bayesian VAR model
%   Detailed explanation goes here

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

% Posterior parameter variance and residuals, [Banbura2010], eq.(7)
U = Y - X*B;
Sig_U = U'*U / (T-p-K*p-1);
Sig_B = kron(U'*U, inv(X'*X));

% Parameters w/out intercept
A = B(1:end-1,:)';
% Sig_A = Sig_B(1:end-K,1:end-K);

%% IRFs computation
% Follows from Lutkepohl (2005) and Banbura & al. (2010)

M = 1000;    % Replications from posterior

IRF       = zeros(K^2, H_irf+1);    % Impulse response functions
IRFdraws  = zeros(M, K^2, H_irf+1); % IRF draws from posterior
sIRF      = zeros(K^2, H_irf+1);    % Structural IRFs
sIRFdraws = zeros(M, K^2, H_irf+1); % Structural IRFs draws from posterior

Ac = [A; eye(K*(p-1)), zeros(K*(p-1), K)];   % Companion form

P = chol(Sig_U)';                            % Structural assumption

vec = @(x) x(:);                             % Vec operation

% Point-wise (posterior mean) impulse responses
IRF(:,1)  = vec(eye(K));
sIRF(:,1) = vec(P);

Ac_pow = zeros(K*p,K*p,H_irf+1);
Ac_pow(:,:,1) = eye(size(Ac));

for h = 1:H_irf
    Ac_pow(:,:,h+1) = Ac*Ac_pow(:,:,h);
    Phi_i = Ac_pow(1:K,1:K,h+1);      % = J*(Ac^h)*J';
    
    IRF(:,h+1)  = vec(Phi_i);
    sIRF(:,h+1) = vec(Phi_i*P);
end

% Replication of impulse responses from posterior draws
% Follows from Banbura & al. (2010), pp 82-83
H = inv(X'*X);
S0 = U'*U;
[~, D0] = iwishrnd(S0, K+2+T);
for m = 1:M
    % Draw from noise variance posterior, [Banbura2010] eq.(7)
    Psi_b = iwishrnd(S0, K+2+T, D0);      % adjust mean  
    % Draw from parameters posterior
    B_b = mvnrnd(vec(B), kron(Psi_b, H));
    B_b = reshape(B_b, K*p+1, K);
    A_b = B_b(1:K*p,:)';
    
    Ac_b = [A_b; eye(K*(p-1)), zeros(K*(p-1), K)];
    P = chol(Psi_b)';
    
    % Compute impulse reponses
    IRFdraws(m,:,1)  = vec(eye(K));
    sIRFdraws(m,:,1) = vec(P);
    Ac_pow_b = zeros(K*p,K*p,H_irf+1);
    Ac_pow_b(:,:,1) = eye(size(Ac_b));
    for h = 1:H_irf
        Ac_pow_b(:,:,h+1) = Ac_b*Ac_pow_b(:,:,h);
        Phi_i_b = Ac_pow_b(1:K,1:K,h+1);    
        IRFdraws(m,:,h+1)  = vec(Phi_i_b);
        sIRFdraws(m,:,h+1) = vec(Phi_i_b*P);
    end
end
end