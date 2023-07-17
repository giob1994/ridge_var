function [irf, sirf] = bvarIrfs_mod(beta, sigma, nshock, hmax)
% computes IRFs using cholesky ordering
% to shock in position nshock
% up to hosizon hmax
% based on beta and sigma

%   Original version:       Giannone, Lenza and Primiceri (2015)
%                           "Prior Selection for Vector Autoregressions"


[k,n] = size(beta);

lags = (k-1)/n;

%% IRFs at the posterior mode
% -----------------------------------------------------
cholVCM = chol(sigma)';

Y_red = zeros(lags+hmax,n);     % reduced-form IRFs
Y_str = zeros(lags+hmax,n);     % structural IRFs

in = lags;
vecshock = zeros(n,1); 
vecshock(nshock) = 1;
for tau = 1:hmax
    xT_red = reshape(Y_red((in+tau-1:-1:in+tau-lags),:)',k-1,1)';
    xT_str = reshape(Y_str((in+tau-1:-1:in+tau-lags),:)',k-1,1)';
    theta_red = xT_red*beta(2:end,:);
    theta_str = xT_str*beta(2:end,:);
    Y_red(in+tau,:) = theta_red + (tau==1)*(vecshock)';
    Y_str(in+tau,:) = theta_str + (tau==1)*(cholVCM*vecshock)';
end

irf = Y_red(in+1:end,:);
sirf = Y_str(in+1:end,:);

end