function [L_opt, CV_opt]= lambda_CV_ridgeVAR_GLS(Yt, p, F, type)
%lambda_CV_ridgeVAR_GLS: Cross-validate general Lambda RidgeVAR_GLS parameter
%   Use F folds over the sample Yt.
%
%   For reference, see Bergmeir et al. (2018) "A note on the validity of
%   cross-validation for evaluating autoregressive time series prediction"
%

%% Regression matrices
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
vcat = @(V) V(:);
folds = cell(1,F);
for i = 1:F
    f_i = Pr(M*(i-1)+1:min(M*i,T-p));
    f_j = Pr([1:M*(i-1),min(M*i,T-p)+1:end]);
    folds{i}.Z_i = Z(:,f_i);
    folds{i}.Y_i = Y(:,f_i);
    folds{i}.Vz_j = kron( Z(:,f_j)*Z(:,f_j)', Su_inv );
    folds{i}.Dz_j = kron( Z(:,f_j), Su_inv ) * vcat(Y(:,f_j));
end

%% Error function
Er = @(L) error(folds, K, p, F, L * T);


%% Optimization
% opts = optimoptions('fmincon', 'Algorithm', 'active-set', ...
%                             'Display', 'notify'); % Optim options
% [L_opt, CV_opt] = ...
%     fmincon(Er, ones(p,1), [], [], [],[], ...
%                                     zeros(p,1), 10^2*ones(p,1), [], opts);

opts = optimoptions('patternsearch', 'Display', 'none'); % Optim options
[L_opt, CV_opt] = ...
    patternsearch(Er, ones(p,1), [], [], [],[], ...
                                  zeros(p,1), 10^2*ones(p,1), [], opts);
                              
L_opt = L_opt * T; 

end

%%% ---

function e = error(folds, K, p, F, Lambda)
Lmat = sparse(diag(kron([0; kron(Lambda(:), ones(K,1))], ones(K,1))));
e = 0;
for i = 1:F
    Bjl = (folds{i}.Vz_j + Lmat) \ folds{i}.Dz_j;
    Bjl = reshape(Bjl, K, K*p+1);
    e = e + sum((folds{i}.Y_i - Bjl * folds{i}.Z_i).^2, 'all');
end
end

% #####