function [lambda] = CV_SmoothLP(Yt, q, r, Lambda, H_irf, which, cont, folds)
%IRFs_LP Impulse response function estimation by *Smooth* Local Projection
%   See Barnichon and Brownlees (2019)
%
% Inputs:
%           Yt          Input (vector) time series
%           q           Number of lags to use as controls
%           r           Order (r-1) is order of polynomial to shrink towards
%           Lambda      Vector of smoothing parameters to cross-validate
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
%           folds       Number of CV folds to use
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
    error("Please select IR function to validate using 'which'")
else
    k = which(1);
    j = which(2);
end
    
%% Make B-Splines

B = bspline((H_min:H_max)', H_min, H_max+1, H_max+1-H_min, 3);
M = size(B, 2);
    
%% Smooth LP Cross-validation
% Follows from Barnichon and Brownlees (2019)

% Run LP regression equation-by-equation  
Y_k = nan(H_s*(T-H_min), 1);
for t = 1:T-H_min
    idx_t = ((t-1)*H_s+1):(t*H_s);
    idx_y = (t+H_min):min((t+H_max), T);
    
    Y_k(idx_t) = [Yt(k,idx_y)'; nan(H_s-length(idx_y),1)];
end

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

% Cross-validate the smoothing parameter
rss = zeros(1, length(Lambda));
for l = 1:length(Lambda)
    S = X_kj / (X_kj'*X_kj + Lambda(l)*P);
    Q = sum(S .* X_kj, 2);
    rss(l) = sum(((Y_kj - S * (X_kj'*Y_kj)) ./ (1 - Q)).^2);
end
[rss_cv, idx_cv] = min(rss);
lambda = Lambda(idx_cv);

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
