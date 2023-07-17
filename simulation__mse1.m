%
%   This version: Oct 2021
%

% Compare the MSE of different methods using the VARMA process empirically 
% estimated in Kilian and Kim (2011).
%

clear

seed = 13584225;

addpath(...
    [cd '/_Bayes'], [cd '/_Bayes/subroutines'], ...
    [cd '/_Hansen_Stein'], ...
    [cd '/_LP'], ...
    [cd '/_RidgeVAR'], ...
    [cd '/_Utility'], ...
    [cd '/_VAR'] ...
)

% Write-to-LaTeX flag
FLAG_write_latex = true;

% Sample size
T = 200;

% Lags specification
p = 10;

% Length of IRFs
H_irf = 24;

% Monte Carlo replications
B = 10000;

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

%% 

options.model   = @(T_) simVARMA_p_1(A1, M1, Sigma_u, T_);
options.seed    = seed;
options.B       = B;
options.T       = T;
options.H       = H_irf;
options.alpha   = 0.95;
options.type    = 'structural';
options.IRF     = @(H_) varmaIRF(A1, M1, P, H_);

% Choose IRF methods
methods = {};

methods{1}.desc = 'ridge cv';
methods{1}.int  = false;
methods{1}.fun  = @(Y_, H_) irf_do(Y_, H_, 'ridge_cv', p, p, 5);

methods{2}.desc = 'ridge gls cv';
methods{2}.int  = false;
methods{2}.fun  = @(Y_, H_, alpha_) irf_do(Y_, H_, 'ridge_gls_cv', p, p, 5);

methods{3}.desc = 'bvar';
methods{3}.int  = true;
methods{3}.fun  = @(Y_, H_, alpha_) irf_do(Y_, H_, 'bvar_cv', p, 0);

methods{4}.desc = 'glp';
methods{4}.int  = true;
methods{4}.fun  = @(Y_, H_, alpha_) irf_do(Y_, H_, 'bvar_glp', p, 0, false);

methods{5}.desc = 'lp';
methods{5}.int  = false;
methods{5}.fun  = @(Y_, H_) irf_do(Y_, H_, 'lp', p);

methods{6}.desc = 'var';
methods{6}.int  = false;
methods{6}.fun  = @(Y_, H_) irf_do(Y_, H_, 'var', p);

% Choose reference method
ref_method = 6;     % LS VAR

%% Simulation
% Only pointwise IRFs are of interest

K = size(P, 1);
H = options.H;

%M = kron(eye(K), ones(1,K))/K;      % Averaging matrix

% True impulse responses
[IRF_true, sIRF_true] = options.IRF(H);
if strcmp(options.type, 'reduced')
    IRF = IRF_true;
elseif strcmp(options.type, 'structural')
    IRF = sIRF_true;
end

% Create storage object
res = repmat(struct('method', []), length(methods), 1);
for n = 1:length(methods)
    res(n).method = methods{n}.desc;
    res(n).type   = options.type;
    res(n).MSE    = zeros(K^2, H+1);
    res(n).Bias   = zeros(K^2, H+1);
end

tic 
for b = 1:B
% parfor b = 1:B
    % Simulate model
    Yt = options.model(options.T);
    Yt = Yt - mean(Yt, 2);

    % Construct IRFs and CIs
    for n = 1:length(methods)
        if ~methods{n}.int      % method returns asymptotic variances
            irfobj = methods{n}.fun(Yt, H);
            if strcmp(options.type, 'reduced')
                IRF_est = irfobj.IRF;
                Std_est = irfobj.Std;
            elseif strcmp(options.type, 'structural')
                IRF_est = irfobj.sIRF;
                Std_est = irfobj.sStd;
            end
            
            res(n).MSE    = res(n).MSE + (IRF - IRF_est).^2/B;
            res(n).Bias   = res(n).Bias + (IRF - IRF_est)/B;
            
        else                    % method returns CIs
            irfobj = methods{n}.fun(Yt, H, options.alpha);
            if strcmp(options.type, 'reduced')
                IRF_est = irfobj.IRF;
                CIs_est = irfobj.CIs;
            elseif strcmp(options.type, 'structural')
                IRF_est = irfobj.sIRF;
                CIs_est = irfobj.sCIs;
            end
            
            res(n).MSE    = res(n).MSE + (IRF - IRF_est).^2/B;
            res(n).Bias   = res(n).Bias + (IRF - IRF_est)/B;
        end
    end
    
    if mod(b, 10) == 0, fprintf('.'), end
    if mod(b, 100) == 0, fprintf(' %6d \n', b), end
end
fprintf('\n')
toc

% Normalite with respect to reference method
% for n = 1:length(methods)
%     res(n).MSE2ref   = sqrt(res(n).MSE ./ res(ref_method).MSE);
%     res(n).Bias2ref  = (res(n).Bias ./ res(ref_method).Bias);
% end

for n = 1:length(methods)
    n_MSE2ref  = zeros(K, H+1);
    n_Bias2ref = zeros(K, H+1);
    for k = 1:K
        k_idx = (k-1) + (1:K:K^2);
        n_MSE2ref(k,:) = sum(res(n).MSE(k_idx,:), 1) ./ sum(res(ref_method).MSE(k_idx,:), 1);
        n_Bias2ref(k,:) = sum(res(n).Bias(k_idx,:), 1) ./ sum(res(ref_method).Bias(k_idx,:), 1);
    end    
    res(n).MSE2ref   = n_MSE2ref;
    res(n).Bias2ref  = n_Bias2ref;
end

%% Show one realization of the methods

% options.CIs = false;

% irf_show(2, options, methods, [])
% irf_show(5, options, methods, [])
% irf_show(9, options, methods, [])

%% Table

% MSE
tab = [];
for j = 1:K
    tab = [tab; ...
        res(1).MSE2ref(j, 1+[1, 4:4:24]); ...
        res(2).MSE2ref(j, 1+[1, 4:4:24]); ...
        res(3).MSE2ref(j, 1+[1, 4:4:24]); ...
        res(4).MSE2ref(j, 1+[1, 4:4:24]); ...
        res(5).MSE2ref(j, 1+[1, 4:4:24]); ...
    ];
end
tab = [kron(1:K, ones(1,5))', tab];

MSE_Table = array2table(tab);
Names = {};
for n = 1:length(methods)-1
    Names(n,1) = {methods{n}.desc};
end
MSE_Table = [repmat(Names, K, 1), MSE_Table];    

MSE_Table.Properties.VariableNames = [...
    {'Estimator'}, ...
    {'IR'}, ...
    arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false) ...
 ];

disp(MSE_Table)

% LaTeX table
if FLAG_write_latex
    LaTeX_MSE_tab.data = tab(:,2:end);
    LaTeX_MSE_tab.tableColLabels = ...
        arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false);
    LaTeX_MSE_tab.tableRowLabels = repmat(Names', 1, K);
    LaTeX_MSE_tab.dataFormat = {'%.2f'};
    LaTeX_MSE_tab.tableCaption = 'IR Estimates - Root MSE Relative to OLS';
    LaTeX_MSE_tab.tableLabel = 'tab:ir_rootmse_1';
    LaTeX_MSE_tab.tableBorders = false;
    LaTeX_MSE_tab.booktabs = true;
    
    fid = fopen('simulation__mse1.tex', 'w');
    latex = latexTable(LaTeX_MSE_tab);
    % save LaTex code as file
    [nrows, ncols] = size(latex);
    for row = 1:nrows
        fprintf(fid, '%s\n', latex{row,:});
    end
    fclose(fid);
end

%% Save results

save(strcat('/pfs/data5/home/ma/ma_ma/ma_gballari/MATLAB/ridge_var/', ...
        "C__simulation_mse1_", ...
        getenv('SLURM_JOBID'), '.mat'));

    
% #####