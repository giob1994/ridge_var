function [IRF,Std] = IRFs_SmoothLP(Yt, q, r, lambda, H_irf, which, cont)
%IRFs_LP Impulse response function estimation by *Smooth* Local Projection
%   See Barnichon and Brownlees (2019)
%
% Inputs:
%           Yt          Input (vector) time series
%           q           Number of lags to use as controls
%           r           Order (r-1) is order of polynomial to shrink towards
%           lambda      Smoothing parameter
%           H_irf       Vector of IRF horizons to compute (note that if
%                       H_irf = [0:H] this implies no contemporaneous
%                       restrictions on IRF, but H_irf = [1:H] implies no
%                       contemporaneous impact)
%           which       Vector either [] (compute ALL impulse responses
%                       according to horizon H_irf restrictions, NOT SUGGESTED),
%                       or [i, j] (impact of j-th variable of i-th variable).
%           cont        Vector of contemporaneous controls to use in
%                       estimating the impulse response, necessary for
%                       recursive identification.
%

T = size(Yt, 2);
K = size(Yt, 1);

if length(H_irf) > 1
    H_min = min(H_irf);
    H_max = max(H_irf);
else
    H_min = 0;
    H_max = H_irf;
end
H_s = H_max+1-H_min;

if isempty(which) || length(which) ~= 2
    which_k = 1:K;
    which_j = 1:K;
    which   = nan;
    cont    = nan;       % if variables are not specified then force 
                        % controls to be undefined
else
    which_k = which(1);
    which_j = which(2);
end
    
%% Make B-Splines

B = bspline((H_min:H_max)', H_min, H_max+1, H_max+1-H_min, 3);
M = size(B, 2);
    
%% Smooth LP computation
% Follows from Barnichon and Brownlees (2019)

if isnan(which)
    IRF = zeros(K^2, H_max+1);    % Impulse response functions
    Std = zeros(K^2, H_max+1);    % Stdev. of IRFs
else
    IRF = zeros(1, H_max+1);
    Std = zeros(1, H_max+1);
end

% Run LP regression equation-by-equation
for k = which_k    
    Y_k = nan(H_s*(T-H_min), 1);
    for t = 1:T-H_min
        idx_t = ((t-1)*H_s+1):(t*H_s);
        idx_y = (t+H_min):min((t+H_max), T);
        
        Y_k(idx_t) = [Yt(k,idx_y)'; nan(H_s-length(idx_y),1)]; 
    end    

    for j = which_j
        x_j = Yt(j,:)';             % Regressors of interest
        if isnan(cont)
            cont_idx = [1:(j-1),(j+1):K];
        else
            cont_idx = cont;
        end
        z_j = [ones(T,1), Yt(cont_idx, :)', lagmatrix(Yt', 1:q)]; 
        z_j(~isfinite(z_j)) = 0;    % Other lags
        
        Xb_j = kron(x_j(1:T-H_min), B);
        Xc_j = kron(z_j(1:T-H_min,:), eye(H_s));
        X_j  = [Xb_j, Xc_j];
        
        sel = isfinite(Y_k);
        Y_kj = Y_k(sel);
        X_kj = sparse(X_j(sel,:));
        
        % Penalization matrix
        P = zeros(size(X_kj, 2));
        D = eye(M);
        for o = 1:r
            D = diff(D);
        end
        P(1:M, 1:M) = D'*D;
        
        delta = std(x_j - z_j/(z_j'*z_j)*z_j'*x_j);
        
        % Compute SmoothLP coefficients
        theta = (X_kj'*X_kj + lambda*P)\(X_kj'*Y_kj);
        
        % Compute variance using Newey-West
        U_kj = Y_kj - X_kj * theta;     % residuals
        
        V_kj = zeros(length(theta));
        weights = [0.5, (H_max+1-(0:H_max))/(H_max+1)];
        
        S_kj = X_kj .* U_kj;
        
        F = kron(eye(T-H_min), ones(H_s,1));
        Uf_ = U_kj .* F(sel, :);
        
        V_kj  = S_kj' * S_kj;
        for l = 0:H_max             
            % % tic
            % idx_t_   = l*H_s+1:(T-H_s-1)*H_s;
            % idx_t_l_ = 1:(T-H_s-1-l)*H_s;
            % S1_ = X_kj(idx_t_, :)' * Uf_(idx_t_, l+1:(T-H_s-1));
            % S2_ = X_kj(idx_t_l_, :)' * Uf_(idx_t_l_, 1:(T-H_s-1-l));
            % GGp_ = S1_*S2_' + S2_*S1_';
            % % toc
            
            % V_kj = V_kj + weights(l+1) * GGp_;
            
            if l > 0
                Gamma_kj_i = S_kj((l+1):T,:)' * S_kj(1:(T-l),:);
                V_kj = V_kj + weights(l)*(Gamma_kj_i + Gamma_kj_i');
            end
        end
        
        Bread = X_kj'*X_kj + (lambda/10)*P;
        Var   = (Bread \ V_kj) / Bread; 
        
        % Store
        if isnan(which)
            % IRF
            IRF(K*(j-1)+k, (1+H_min):end) = B * theta(1:M) * delta;
            % Std
            Std(K*(j-1)+k, (1+H_min):end) = sqrt(diag(B*Var(1:M, 1:M)*B')) * delta;
        else
            % IRF
            IRF((1+H_min):end) = B * theta(1:M) * delta;
            % Std
            Std((1+H_min):end) = sqrt(diag(B*Var(1:M, 1:M)*B')) * delta;
        end
    end
end

end

% B-Spline function
%   Taken from Barnichon and Brownlees (2019) original code
function B_ = bspline(x, xl, xr, ndx, bdeg)
    dx = (xr - xl) / ndx;
    t = xl + dx * (-bdeg:ndx-1);
    T_ = (0 * x + 1) * t;
    X_ = x * (0 * t + 1);
    P_ = (X_ - T_) / dx;
    B_ = (T_ <= X_) & (X_ < (T_ + dx));
    r = [2:length(t) 1];
    for k_ = 1:bdeg
        B_ = (P_ .* B_ + (k_ + 1 - P_) .* B_(:, r)) / k_;
    end
end
