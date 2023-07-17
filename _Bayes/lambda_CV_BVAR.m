function [L_opt, CV_opt]= lambda_CV_BVAR(Yt, p, epsilon, F, type)
%lambda_CV_BVAR: Cross-validate general Lambda BVAR parameter (as if Ridge)
%   Use F folds over the sample Yt.
%
%   For reference, see Bergmeir et al. (2018) "A note on the validity of
%   cross-validation for evaluating autoregressive time series prediction"
%

%% Regression matrices
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

%% Build folds
M = floor((T-p) / F);
switch type
    case 'shuffle'
        Pr = randperm(T-p);
    case 'block'
        Pr = 1:(T-p);
    case 'block_nondep'
        M = floor((T-F*p) / F);
        Pr = zeros(1, F*M);
        for i = 1:F
            Pr(1+M*(i-1):min(M*i,T-F*p)) = (1+(M+p)*(i-1)):(M*i+p*(i-1));
        end
    otherwise
        error("Unrecognized CV fold structure")
end

%% Storing the folds before optimizations for faster execution
folds = cell(1,F);
for i = 1:F
    f_i = Pr(M*(i-1)+1:min(M*i,T-p));
    f_j = Pr([1:M*(i-1),min(M*i,T-p)+1:end]);
    folds{i}.X_i = X(f_i,:);
    folds{i}.Y_i = Y(f_i,:);
    folds{i}.X_j = X(f_j,:);
    folds{i}.Y_j = Y(f_j,:);
end

%% Error function
Er = @(L) error(folds, sigma, epsilon, K, p, F, L * T);


%% Optimization
% opts = optimoptions('fmincon', 'Algorithm', 'active-set', ...
%                             'Display', 'notify'); % Optim options
% [L_opt, CV_opt] = ...
%     fmincon(Er, ones(p,1), [], [], [],[], ...
%                                     zeros(p,1), 10^4*ones(p,1), [], opts);

opts = optimoptions('patternsearch', 'Display', 'none'); % Optim options
[L_opt, CV_opt] = ...
    patternsearch(Er, 1, [], [], [],[], ...
                                  1e-5, 10^2, [], opts);
                              
L_opt = L_opt * T; 

end

%%% ---

function e = error(folds, sigma, epsilon, K, p, F, lambda)
e = 0;
Yd = [ diag(sigma)/lambda; zeros(K*(p-1), K); diag(sigma); zeros(1, K) ];
Xd = [ kron(diag(1:p), diag(sigma)/lambda), zeros(K*p, 1);
       zeros(K, K*p+1);
       zeros(1, K*p), epsilon ];
for i = 1:F
    Y_j = [folds{i}.Y_j; Yd];
    X_j = [folds{i}.X_j; Xd];
    Bjl = (X_j' * X_j) \ X_j' * Y_j;
    e = e + sum((folds{i}.Y_i - folds{i}.X_i * Bjl).^2, 'all');
end
end

% #####