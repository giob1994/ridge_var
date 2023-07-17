%
%
%

clear

seed = 13584225;

% SAVE PLOTS?
save_flag = true;

set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

addpath(...
    [cd '/_Bayes'], ...
    [cd '/_RidgeVAR'], ...
    [cd '/_Utility'], ...
    [cd '/_VAR'] ...
)

% VAR(2)
A     = [  0.8, 0.1,   0.1, -0.2;
          -0.1, 0.7,     0,  0.1; ];
Sigma = [  0.3,   0;  
             0,   5; ];

fprintf("Stationary = %d\n\n", stationary(A))
  
rng(seed)
Yt = simVAR_p(A, Sigma, 200);
Yt = Yt - mean(Yt, 2);

%%

% % Ridge VAR
% L_ridge = lambda_CV_ridgeVAR(Yt, 3, 5, 'block');
% A_ridge = Ridge_VAR(Yt, 3, L_ridge);
% 
% % Ridge VAR GLS
% L_rgls = lambda_CV_ridgeVAR_GLS(Yt, 3, 5, 'block');
% A_rgls = Ridge_VAR_GLS(Yt, 3, L_rgls);
% 
% % BVAR
% L_bvar = lambda_CV_BVAR(Yt, 3, 1e-3, 5, 'block');
% A_bvar = BVAR(Yt, 3, L_bvar, 1e-3);

%%

% p = 3;
% T = size(Yt, 2);
% K = size(Yt, 1);
% 
% Z = zeros(K*p, T-p);
% for i = 1:p
%     Z((1+(i-1)*K):(i*K),:) = Yt(:,(p+1-i):(T-i));
% end
% Z_pre = Z;
% Z = [ones(1, T-p); Z]; % add intercept
% Y = Yt(:,(p+1):T);
% 
% L1 = diag([0, kron([0, 1,  0.1], [1, 1])]);
% L2 = diag([0, kron([0, 100, 0.1], [1, 1])]);
% 
% B1 = Y * Z' / (Z * Z' + diag([0, kron([100, 1,  0.1], [1, 1])]));
% B2 = Y * Z_pre' / (Z_pre * Z_pre' + kron([100, 1,  0.1], [1, 1]));
% 
% mean1 = B1(:,1);
% mean2 = mean(Y, 2) - B2 * mean(Z_pre, 2);
% 
% R1 = Z*Z' + L1;
% R2 = Z*Z' + L2;

% R2_inv = inv(R2);
% 
% R2_inv * R1
% 
% inv(eye(1+2*p) + R1 \ (L2 - L1))
% 
% norm(R2_inv * R1)

%% Plot Max Eigenval over 2D

p = 2;
M = 150;
lambdas = logspace(-2, 4, M);

Eig_grid = zeros(M, M);
for m1 = 1:M
    for m2 = 1:M
        A_hat = Ridge_VAR(Yt, p, [lambdas(m1), lambdas(m2)]);
        Ac_j = [A_hat; eye(2*(p-1)), zeros(2*(p-1), 2)];
        Eig_grid(m2, m1) = abs(eigs(Ac_j, 1));
    end
end
A_LS = LS_VAR(Yt, p);
Ac_j = [A_LS; eye(2*(p-1)), zeros(2*(p-1), 2)];
Eig_LS = abs(eigs(Ac_j, 1));

levels = [0.95, 0.9, 0.875, 0.85, 0.825, 0.8, ...
            0.75, 0.775, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2];

fig1 = figure;
colormap('gray')
[L1, L2] = meshgrid(log10(lambdas), log10(lambdas));
contourf(L1, L2, Eig_grid, levels, 'ShowText', 'on');
hold on
% LS
scatter(log10(lambdas(1)), log10(lambdas(1)), 20, 'k', 'filled')
text(log10(lambdas(1))+0.1, log10(lambdas(1))+0.2, ...
        sprintf("LS: %2.3f", Eig_LS))
grid on
xlabel("$\lambda_1$")
ylabel("$\lambda_2$")
xt = xticks;
xticklabels(arrayfun(@(x) sprintf('$10^{%d}$', x), xt, 'UniformOutput', false));
yt = yticks;
yticks(yt(2:end));
yticklabels(arrayfun(@(x) sprintf('$10^{%d}$', x), yt(2:end), 'UniformOutput', false));
ax = gca;
ax.FontSize = 12;


% Save
if save_flag
    disp('Saving plot...')
    set(fig1, 'Units', 'inches', 'Position', [1, 1, 4.5, 3.5]);
    printpdf(fig1, 'ridge_shrink__stationary.pdf')
end

%% Estimate Shrink Ridge

p = 2;

%lambdas = linspace(1e-2, 1000, 50);
lambdas = logspace(-2, 6, 30);

A_hat = zeros(2, 2*p, length(lambdas));
for j = 1:length(lambdas)
    L = lambdas(j) * [1, 0];
    A_hat(:,:,j) = Ridge_VAR(Yt, p, L);
    %A_hat(:,:,j) = Ridge_VAR_GLS(Yt, p, L);
    %BVAR(Yt, p, lambdas(j), 1e-3);
end

% Estimate LS submodel
A_hat_LS = LS_VAR(Yt, 2);

% Norm
%A_hat_eig    = zeros(1, length(lambdas));
A_hat_norm   = zeros(1, length(lambdas));
A_hat_p_norm = zeros(p, length(lambdas));
for j = 1:length(lambdas)
    %Ac_j = [squeeze(A_hat(:,:,j)); eye(2*(p-1)), zeros(2*(p-1), 2)];
    %A_hat_eig(j)  = abs(eigs(Ac_j, 1));
    A_hat_norm(j) = norm(squeeze(A_hat(:,:,j)), 'fro');
    for k = 1:p
        A_hat_p_norm(k,j) = norm(squeeze(A_hat(:,(1+(k-1)*2):k*2,j)), 'fro');
    end
end

% Difference
% A_hat_diff = zeros(1, length(lambdas));
% for j = 1:length(lambdas)
%     A_hat_diff(j) = norm(([A_hat_LS, zeros(2, (p-1))] - squeeze(A_hat(:,:,j))));
% end

% Plot
fig2 = figure;
p0 = semilogx(lambdas, A_hat_norm, 'k', 'Marker', '.');
hold on
p1 = semilogx(lambdas, A_hat_p_norm(1,:), 'k', 'Marker', 'o', 'MarkerSize', 5);
p2 = semilogx(lambdas, A_hat_p_norm(2,:), 'k', 'Marker', '*', 'MarkerSize', 5);
hold off
l0 = yline(norm(A_hat_LS, 'fro'), 'k--', 'LabelHorizontalAlignment', 'left', 'LineWidth', 1.3);
ylim([0, 1.2])
grid
xlabel("$\lambda_1$")
legend([l0, p0, p1, p2], ...
    ["$||\hat{B}^{LS}||_F$", "$||\hat{B}^{R}(\Lambda^{(2)})||_F$", ...
            "$||\hat{A}^{R}_1(\Lambda^{(2)})||_F$", ...
            "$||\hat{A}^{R}_2(\Lambda^{(2)})||_F$"], ...
    'Location', "west")
ax = gca;
ax.FontSize = 12;

% Save
if save_flag
    disp('Saving plot...')
    set(fig2, 'Units', 'inches', 'Position', [1, 1, 4.5, 3.5]);
    printpdf(fig2, 'ridge_shrink__norm.pdf')
end

%% Plot Bias over 2D

p = 2;
M = 150;
lambdas = logspace(-2, 4, M);

Bias_grid = zeros(M, M);
for m1 = 1:M
    for m2 = 1:M
        A_hat_RG = Ridge_VAR(Yt, p, [lambdas(m1), lambdas(m2)]);
        Bias_grid(m2, m1) = norm(A_hat_RG - A, Inf);
    end
end
Bias_LS = norm(LS_VAR(Yt, p) - A, Inf);

levels = [1.6, 1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 0.16, 0.14, 0.12];

% Find minimum bias
[Bias_min, idx] = min(Bias_grid, [], 'all', 'linear');
idx_1 = sum(Bias_grid == min(Bias_grid, [], 'all'), 1) == 1;
idx_2 = sum(Bias_grid == min(Bias_grid, [], 'all'), 2) == 1;

fig3 = figure;
cmap = colormap('gray');
colormap(cmap(150:end,:))
set(groot,'defaultAxesTickLabelInterpreter','latex');  
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
[L1, L2] = meshgrid(log10(lambdas), log10(lambdas));
contourf(L1, L2, Bias_grid, levels, 'ShowText', 'on');
% contourf(L1, L2, Bias_grid, 'ShowText', 'on');
hold on
% Minimum
scatter(log10(lambdas(idx_1)), log10(lambdas(idx_2)), 20, 'filled', ...
            'MarkerFaceColor', 	'b')
text(log10(lambdas(idx_1))-1.9, log10(lambdas(idx_2))+0.3, ...
        sprintf("Minimum: %2.3f", Bias_min), 'Color', 'b')
% LS
scatter(log10(lambdas(1)), log10(lambdas(1)), 20, 'k', 'filled')
text(log10(lambdas(1))+0.1, log10(lambdas(1))+0.2, ...
        sprintf("LS: %2.3f", Bias_LS))
grid on
xlabel("$\lambda_1$")
ylabel("$\lambda_2$")
xt = xticks;
xticklabels(arrayfun(@(x) sprintf('$10^{%d}$', x), xt, 'UniformOutput', false));
yt = yticks;
yticks(yt(2:end));
yticklabels(arrayfun(@(x) sprintf('$10^{%d}$', x), yt(2:end), 'UniformOutput', false));
ax = gca;
ax.FontSize = 12;


% Save
if save_flag
    disp('Saving plot...')
    set(fig3, 'Units', 'inches', 'Position', [1, 1, 4.5, 3.5]);
    printpdf(fig3, 'ridge_shrink__bias.pdf')
end

% #####