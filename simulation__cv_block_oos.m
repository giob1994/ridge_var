%
%   This version: Oct 2021
%

% Give a plot of the block CV/OOS validated ridge lag-adapted
% regularizers that are selected for the GLP (2015) dataset.

clear

seed = 13584225;

% SAVE PLOTS?
save_flag = true;

addpath(...
    [cd '/_RidgeVAR'], ...
    [cd '/_Utility'], ...
    [cd '/_VAR'] ...
)

% Lags specification
p = 10;

% Length of IRFs
H_irf = 24;

% Monte Carlo replications
B = 100;

% VARMA matrices from Kilian & Kim (2011)
A1 = [  0.5417 -0.1971 -0.9395;
          0.04    0.9677  0.0323;
         -0.0015  0.0829  0.8080  ];
M1 = [ -0.1428 -1.5133 -0.7053;
         -0.0202  0.0309  0.1561;
          0.0227  0.1178 -0.0153  ];
      
P = [  9.2325  0.0     0.0   ;
      -1.4343  3.6070  0.0   ;
      -0.7756  1.2296  2.7555  ];  
% P = eye(K);

Sigma_u = P*P';

%% Simulation

mat_lambda_CV5 = nan(B, p);
mat_lambda_CV10 = nan(B, p);
mat_lambda_BNDCV = nan(B, p);
mat_lambda_OOS = nan(B, p);

tic 
parfor b = 1:B
    % Simulate model
    Yt = simVARMA_p_1(A1, M1, Sigma_u, 200);
    Yt = Yt - mean(Yt, 2);
    
    % CV5 (5 folds)
    [lambda_CV5, ~] = lambda_CV_ridgeVAR(Yt, p, 5, 'block');
    
    % CV10 (10 folds)
    [lambda_CV10, ~] = lambda_CV_ridgeVAR(Yt, p, 10, 'block');
    
    % Block non-dependent CV 
    [lambda_BNDCV, ~] = lambda_CV_ridgeVAR(Yt, p, 10, 'block_nondep');
    
    % OOS
    [lambda_OOS, ~] = lambda_OOS_ridgeVAR(Yt, p, 0.2);
    
    mat_lambda_CV5(b,:)   = lambda_CV5;
    mat_lambda_CV10(b,:)  = lambda_CV10;
    mat_lambda_BNDCV(b,:) = lambda_BNDCV;
    mat_lambda_OOS(b,:)   = lambda_OOS;
    
    fprintf('b = %6d\n', b)
end
toc

%% Plot

mlambda_CV5   = mean(mat_lambda_CV5);
mlambda_CV10  = mean(mat_lambda_CV10);
mlambda_BNDCV = mean(mat_lambda_BNDCV);
mlambda_OOS   = mean(mat_lambda_OOS);

fig1 = figure();
bar([mlambda_CV5(:), mlambda_CV10(:), mlambda_BNDCV(:), mlambda_OOS(:)])
xlabel('Lag')
ylabel('Average \lambda_i')
grid on
set(gca, 'YScale', 'log')
yline(20000, 'k--')
ylim([10, 30000])
legend({'CV 5', 'CV 10', 'BND CV 10', 'OS'}, 'Location', 'southeast')

% Save
if save_flag
    disp('Saving plot...')
    set(fig1, 'Units', 'inches', 'Position', [1, 1, 4.5, 3.5]);
    printpdf(fig1, 'ridge_cv_blocl_oos_avg.pdf')
end

%%
fig2 = figure();
boxplot(mat_lambda_CV10 + 10^-9, 'BoxStyle', 'filled', 'Colors', 'k', 'Symbol', 'k.')
xlabel('Lag')
ylabel('\lambda_i')
grid on
set(gca, 'YScale', 'log')
yline(20000, 'k--')
ylim([10, 30000])

% Save
if save_flag
    disp('Saving plot...')
    set(fig2, 'Units', 'inches', 'Position', [1, 1, 4.5, 3.5]);
    printpdf(fig2, 'ridge_cv10_boxplot.pdf')
end
