function [L_opt, CV_opt]= lambda_OOS_ridgeVAR(Yt, p, D)
%lambda_OOS_ridgeVAR Out-Of-Sample validate general Lambda RidgeVAR parameter
%   Use D*(T-p) observations at the end of sample
%
%   For reference, see Bergmeir et al. (2018) "A note on the validity of
%   cross-validation for evaluating autoregressive time series prediction"
%

T = size(Yt, 2);
K = size(Yt, 1);

Z = zeros(K*p, T-p);
for i = 1:p
    Z((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
end
Y = Yt(:,(p+1):T);

F = T-floor(D*(T-p))-1; % [1:F] index estimation sample

Z_t = Z(:,1:F-p);
Y_t = Y(:,1:F-p);
Z_j = Z(:,F:end);
Y_j = Y(:,F:end);

% error function
Er = @(L) error(Y_t, Z_t, Y_j, Z_j, K, T * L);

% opts = optimoptions('fmincon', 'Algorithm', 'active-set', ...
%                             'Display', 'notify'); % Optim options
% [L_opt, CV_opt] = ...
%     fmincon(Er, ones(p,1), [], [], [],[], ...
%                                     zeros(p,1), 10^2*ones(p,1), [], opts);

opts = optimoptions('patternsearch', 'Display', 'none'); % Optim options
[L_opt, CV_opt] = ...
    patternsearch(Er, ones(p,1), [], [], [],[], ...
                                    zeros(p,1), 10^2*ones(p,1), [], opts);
                                
L_opt = T * L_opt;

end

%%% ---

function e = error(Y_t, Z_t, Y_j, Z_j, K, Lambda)
% Lmat = kron(diag(Lambda), eye(K));
Lmat = diag(kron(Lambda(:), ones(K,1)));
Bjl = Y_t * Z_t' / (Z_t * Z_t' + Lmat);
e = norm((Y_j - Bjl * Z_j), 'fro');
end

% #####