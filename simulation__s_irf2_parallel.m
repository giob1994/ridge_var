%
%   This version: Oct 2021
%

% Simulate a VAR model estimated on the data from Kilian & Kim (2011) with
% recursive identification, then estimate IRFs 
%

clear

FLAG_write_latex = true;

%% Cluster preamble

if ~isempty(getenv('SLURM_JOB_CPUS_PER_NODE'))
    % create a local cluster object
    pc = parcluster('local');
    
    % get the number of dedicated cores from environment
    num_workers = str2num(getenv('SLURM_JOB_CPUS_PER_NODE'));
    fprintf("[Cluster] SLURM_JOB_CPUS_PER_NODE: %d\n\n", num_workers)
    
    % explicitly set the JobStorageLocation to the tmp directory that is unique to each cluster job (and is on local, fast scratch)
    parpool_tmpdir = [getenv('TMP'),'/.matlab/local_cluster_jobs/slurm_jobID_',getenv('SLURM_JOB_ID')];
    mkdir(parpool_tmpdir)
    pc.JobStorageLocation = parpool_tmpdir;
    
    % start the parallel pool
    parpool(pc,num_workers)
end

%% Settings

seed = 13584225;

addpath(...
    [cd '/_Bayes'], [cd '/_Bayes/subroutines'], ...
    [cd '/_Hansen_Stein'], ...
    [cd '/_LP'], ...
    [cd '/_RidgeVAR'], ...
    [cd '/_Utility'], ...
    [cd '/_VAR'] ...
)

% Sample size
T = 200;

% Lags specification
p = 10;

% Length of IRFs
H_irf = 24;

% Monte Carlo replications
if ~isempty(getenv('SLURM_JOB_CPUS_PER_NODE'))
    B = 1000; % for cluster
else
    B = 1; % for local testing
end

%% Simulation model

K = 3;

% VARMA matrices from Kilian & Kim (2011)
A1  = [  0.5417 -0.1971 -0.9395;
          0.04    0.9677  0.0323;
         -0.0015  0.0829  0.8080  ];
M1  = [ -0.1428 -1.5133 -0.7053;
         -0.0202  0.0309  0.1561;
          0.0227  0.1178 -0.0153  ];
P = [  9.2325  0.0     0.0   ;
      -1.4343  3.6070  0.0   ;
      -0.7756  1.2296  2.7555  ];  
% P = eye(K);

Sigma_u = P * P'; 

%% Simulation Options

%options.model   = simVAR_p(A, Sigma_u, T_);
options.seed    = seed;
options.B       = B;
options.T       = T;
options.H       = H_irf;
options.alpha   = 0.9;
options.type    = 'structural';

% Choose IRF methods
methods = {};

methods{1}.desc = 'var';
methods{1}.int  = false;

methods{2}.desc = 'ridge cv';
methods{2}.int  = false;

methods{3}.desc = 'ridge cv a-shrink';
methods{3}.int  = false;

methods{4}.desc = 'bvar cv';
methods{4}.int  = true;

methods{5}.desc = 'glp';
methods{5}.int  = true;

methods{6}.desc = 'lp';
methods{6}.int  = false;

% methods{5}.desc = 'comb stein';
% methods{5}.int  = false;
% methods{5}.fun  = @(Y_, H_) irf_do(Y_, H_, 'comb_stein_hansen', p);

%% Simulation

rng(options.seed);
% parseeds = seed + randperm(B);

H = H_irf;

% True impulse responses
[IRF_true, sIRF_true] = varmaIRF(A1, M1, P, H);
if strcmp(options.type, 'reduced')
    IRF = IRF_true;
elseif strcmp(options.type, 'structural')
    IRF = sIRF_true;
end

if length(methods) < 1
    error("No methods selected to compute IRFs")
end

% Make local variables for parallelization
alpha_        = options.alpha;
qnorm         = norminv(1-((1-alpha_)/2), 0, 1);
type_         = options.type;

N_methods     = length(methods);
methods_desc  = cellfun(@(x) x.desc, methods, 'UniformOutput', false);
methods_int   = cellfun(@(x) x.int, methods);
resmat_Cover  = zeros(N_methods, K^2, H+1);
resmat_Length = zeros(N_methods, K^2, H+1);
resmat_MSE    = zeros(N_methods, K^2, H+1);
resmat_Bias   = zeros(N_methods, K^2, H+1);

tic 
parfor b = 1:options.B
% for b = 1:B
    % Simulate model
    Yt = simVARMA_p_1(A1, M1, Sigma_u, T);
    Yt = Yt - mean(Yt, 2);
    
    % Load variables
    IRF_        = IRF;
    method_     = methods_desc;
    method_int_ = methods_int;
    
    resmat_cover_b  = zeros(N_methods, K^2, H+1);
    resmat_length_b = zeros(N_methods, K^2, H+1);
    resmat_mse_b    = zeros(N_methods, K^2, H+1);
    resmat_bias_b   = zeros(N_methods, K^2, H+1);

    % Construct IRFs and CIs
    for n = 1:N_methods       
        switch method_{n}
            case 'var'
                irfobj = irf_do(Yt, H, 'var', p);
            case 'lp'
                irfobj = irf_do(Yt, H, 'lp', p);
            case 'bvar cv'
                irfobj = irf_do(Yt, H, 'bvar_cv', p, alpha_);
            case 'glp'
                irfobj = irf_do(Yt, H, 'bvar_glp', p, alpha_, true);
            case 'ridge cv'
                irfobj = irf_do(Yt, H, 'ridge_cv', p, p, 5);
            case 'ridge cv a-shrink'
                irfobj = irf_do(Yt, H, 'ridge_varshrink', p, p, 5, floor(p*2/3));
            otherwise
                error("Invalid IRF method")
        end
        
        if ~method_int_(n)        % method returns asymptotic variances
            if strcmp(type_, 'reduced')
                IRF_est = irfobj.IRF;
                Std_est = irfobj.Std;
            elseif strcmp(type_, 'structural')
                IRF_est = irfobj.sIRF;
                Std_est = irfobj.sStd;
            end
            
            Covg_b = zeros(K^2, H+1);
            for k = 1:K^2
                for h = 1:H+1
                    if (IRF_(k,h) >= IRF_est(k,h)-qnorm*Std_est(k,h) && ...
                            IRF_(k,h) <= IRF_est(k,h)+qnorm*Std_est(k,h))
                        Covg_b(k,h) = 1;
                    end
                end
            end
            resmat_cover_b(n,:,:)  = Covg_b/B;
            resmat_length_b(n,:,:) = (2*qnorm*Std_est)/B;
            resmat_mse_b(n,:,:)    = (IRF_ - IRF_est).^2/B;
            resmat_bias_b(n,:,:)   = (IRF_ - IRF_est)/B;
            
        else                    % method returns CIs
            if strcmp(type_, 'reduced')
                IRF_est = irfobj.IRF;
                CIs_est = irfobj.CIs;
            elseif strcmp(type_, 'structural')
                IRF_est = irfobj.sIRF;
                CIs_est = irfobj.sCIs;
            end
            
            Covg_b = zeros(K^2, H+1);
            for k = 1:K^2
                for h = 1:H+1
                    if (IRF_(k,h) >= CIs_est(1,k,h) && ...
                            IRF_(k,h) <= CIs_est(2,k,h))
                        Covg_b(k,h) = 1;
                    end
                end
            end    
            resmat_cover_b(n,:,:)  = Covg_b/B;
            resmat_length_b(n,:,:) = squeeze(CIs_est(2,:,:) - CIs_est(1,:,:))/B;
            resmat_mse_b(n,:,:)    = (IRF_ - IRF_est).^2/B;
            resmat_bias_b(n,:,:)   = (IRF_ - IRF_est)/B;
            
        end
    end
    fprintf('b = %6d\n', b)
    
    resmat_Cover  = resmat_Cover + resmat_cover_b;
    resmat_Length = resmat_Length + resmat_length_b;
    resmat_MSE    = resmat_MSE + resmat_mse_b;
    resmat_Bias   = resmat_Bias + resmat_bias_b;
end
time = toc;

fprintf("Replications:      %d \n", options.B)
fprintf("Simulation time:   %.2f seconds\n\n", time)

% Create storage object
res = repmat(struct('method', []), length(methods), 1);
for n = 1:length(methods)
    res(n).method = methods{n}.desc;
    res(n).type   = options.type;
    n_Cover  = zeros(K, H+1);
    n_Length = zeros(K, H+1);
    for k = 1:K
        k_idx = (k-1) + (1:K:K^2);
        n_Cover(k,:)  = mean(squeeze(resmat_Cover(n,k_idx,:)));
        n_Length(k,:) = mean(squeeze(resmat_Length(n,k_idx,:)));
    end
    res(n).Cover  = n_Cover; %squeeze(resmat_Cover(n,:,:));
    res(n).Length = n_Length; %squeeze(resmat_Length(n,:,:));
    res(n).MSE    = squeeze(resmat_MSE(n,:,:));
    res(n).Bias   = squeeze(resmat_Bias(n,:,:));
end

%% Table

% Coverage
tab = [];
for j = 1:K
    tab = [tab; ...
        res(1).Cover(j, [2, 1+(4:4:24)]); ...
        res(2).Cover(j, [2, 1+(4:4:24)]); ...
        res(3).Cover(j, [2, 1+(4:4:24)]); ...
        res(4).Cover(j, [2, 1+(4:4:24)]); ...
        res(5).Cover(j, [2, 1+(4:4:24)]); ...
        res(6).Cover(j, [2, 1+(4:4:24)]); ...
    ];
end
tab = [kron(1:K, ones(1,length(methods)))', tab];

Cover_Table = array2table(tab);
Names = {};
for n = 1:length(methods)
    Names(n,1) = {methods{n}.desc};
end
Cover_Table = [repmat(Names, K, 1), Cover_Table];    

Cover_Table.Properties.VariableNames = [...
    {'Estimator'}, ...
    {'IR'}, ...
    arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false) ...
 ];

disp(Cover_Table)

% LaTeX table
if FLAG_write_latex
    LaTeX_Cover_tab.data = tab(:,2:end);
    LaTeX_Cover_tab.tableColLabels = ...
        arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false);
    LaTeX_Cover_tab.tableRowLabels = repmat(Names', 1, K);
    LaTeX_Cover_tab.dataFormat = {'%.2f'};
    LaTeX_Cover_tab.tableCaption = 'IR Estimates - CI Coverage';
    LaTeX_Cover_tab.tableLabel = 'tab:ir_cover_2';
    LaTeX_Cover_tab.tableBorders = false;
    LaTeX_Cover_tab.booktabs = true;
    
    fid = fopen('simulation__s_irf2_cover.tex', 'w');
    latex = latexTable(LaTeX_Cover_tab);
    % save LaTex code as file
    [nrows, ncols] = size(latex);
    for row = 1:nrows
        fprintf(fid, '%s\n', latex{row,:});
    end
    fclose(fid);
end

% Length
tab = [];
for j = 1:K
    tab = [tab; ...
        res(1).Length(j, [2, 1+(4:4:24)]); ...
        res(2).Length(j, [2, 1+(4:4:24)]); ...
        res(3).Length(j, [2, 1+(4:4:24)]); ...
        res(4).Length(j, [2, 1+(4:4:24)]); ...
        res(5).Length(j, [2, 1+(4:4:24)]); ...
        res(6).Length(j, [2, 1+(4:4:24)]); ...
    ];
end
tab = [kron(1:K, ones(1,length(methods)))', tab];

Length_Table = array2table(tab);
Names = {};
for n = 1:length(methods)
    Names(n,1) = {methods{n}.desc};
end
Length_Table = [repmat(Names, K, 1), Length_Table];    

Length_Table.Properties.VariableNames = [...
    {'Estimator'}, ...
    {'IR'}, ...
    arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false) ...
 ];

disp(Length_Table)

% LaTeX table
if FLAG_write_latex
    LaTeX_Length_tab.data = tab(:,2:end);
    LaTeX_Length_tab.tableColLabels = ...
        arrayfun(@(x) sprintf('$h$ = %d', x), [1, 4:4:24], 'UniformOutput', false);
    LaTeX_Length_tab.tableRowLabels = repmat(Names', 1, K);
    LaTeX_Length_tab.dataFormat = {'%.2f'};
    LaTeX_Length_tab.tableCaption = 'IR Estimates - CI Length';
    LaTeX_Length_tab.tableLabel = 'tab:ir_length_2';
    LaTeX_Length_tab.tableBorders = false;
    LaTeX_Length_tab.booktabs = true;
    
    fid = fopen('simulation__s_irf2_length.tex', 'w');
    latex = latexTable(LaTeX_Length_tab);
    % save LaTex code as file
    [nrows, ncols] = size(latex);
    for row = 1:nrows
        fprintf(fid, '%s\n', latex{row,:});
    end
    fclose(fid);
end

%% Save results

save(strcat('/pfs/data5/home/ma/ma_ma/ma_gballari/MATLAB/ridge_var/', ...
        "C__simulation_s_irf2_", ...
        getenv('SLURM_JOBID'), '.mat'));

% #####