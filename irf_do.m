% irf_do()
%   Compute reduced form / structural IRFs using a chosen method
%   Giovanni Ballarin, Aug 2021
%

function [irfobj] = irf_do(Yt, H, method, varargin)
%
%

switch method
    
    % ___________________________________________________
    %   Method: VAR IRFs
    case 'var'
        p = varargin{1};
        % VAR
        [IRF, Std, ~, sIRF, sStd] = IRFs_VAR(Yt, p, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
        
    % ___________________________________________________
    %   Method: Local projections (Newey-West) IRFs
    case 'lp'
        q = varargin{1};
        % Local projection
        [IRF, Std, sIRF,sStd] = IRFs_LP(Yt, q, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
        
    % ___________________________________________________
    %   Method: Ridge-VAR w/ Out-of-Sample validation
    case 'ridge_oos'
        p = varargin{1};
        max_p = p;
        frac = 0.2;
        if length(varargin) > 1
            max_p = varargin{2};
        end
        if length(varargin) > 2
            frac = varargin{3};
        end
        pm = min(p, max_p);
        % Out-of-sample validation
        [lambda, ~] = lambda_OOS_ridgeVAR(Yt, pm, frac);
        lambda = [lambda(:); repmat(lambda(end), p-pm, 1)];
        % Ridge-VAR
        [IRF, Std, ~, sIRF, sStd] = IRFs_ridgeVAR(Yt, p, lambda, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
    
    % ___________________________________________________
    %   Method: Ridge-VAR w/ Cross-validation
    case 'ridge_cv'
        p = varargin{1};
        max_p = p;
        nfolds = 3;
        cv_type = 'block_nondep';
        if length(varargin) > 1
            max_p = varargin{2};
        end
        if length(varargin) > 2
            nfolds = varargin{3};
        end
        if length(varargin) > 3
            cv_type = varargin{4};
        end
        pm = min(p, max_p);
        % Cross-validation
        [lambda, ~] = lambda_CV_ridgeVAR(Yt, pm, nfolds, cv_type);
        lambda = [lambda(:); repmat(lambda(end), p-pm, 1)];
        % Ridge-VAR
        [IRF, Std, ~, sIRF, sStd] = IRFs_ridgeVAR(Yt, p, lambda, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
        
    % ___________________________________________________
    %   Method: Ridge-VAR w/ Cross-validation
    case 'ridge_gls_cv'
        p = varargin{1};
        max_p = p;
        nfolds = 3;
        cv_type = 'block_nondep';
        if length(varargin) > 1
            max_p = varargin{2};
        end
        if length(varargin) > 2
            nfolds = varargin{3};
        end
        if length(varargin) > 3
            cv_type = varargin{4};
        end
        pm = min(p, max_p);
        % Cross-validation
        % [lambda, ~] = lambda_CV_ridgeVAR_GLS(Yt, pm, nfolds, cv_type);
        [lambda, ~] = lambda_CV_ridgeVAR(Yt, pm, nfolds, cv_type);
        lambda = [lambda(:); repmat(lambda(end), p-pm, 1)];
        % Ridge-VAR
        [IRF, Std, ~, sIRF, sStd] = IRFs_ridgeVAR_GLS(Yt, p, lambda, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
        
        
    % ___________________________________________________
    %   Method: Ridge-VAR w/ CV & asymptotic variance shrinkage    
    case 'ridge_varshrink'
        p = varargin{1};
        max_p = p;
        nfolds = 3;
        s = 0;    % add asymtotic shrinkage after the first 's' lags
        cv_type = 'block_nondep';
        if length(varargin) > 1
            max_p = varargin{2};
        end
        if length(varargin) > 2
            nfolds = varargin{3};
        end
        if length(varargin) > 3
            s = varargin{4};
        end
        if length(varargin) > 4
            cv_type = varargin{5};
        end
        pm = min(p, max_p);
        % Cross-validation
        [lambda, ~] = lambda_CV_ridgeVAR(Yt, pm, nfolds, cv_type);
        lambda = [lambda(:); repmat(lambda(end), p-pm, 1)];
        % Ridge-VAR w/ asymptotic variance shrinkage
        [IRF, Std, ~, sIRF, sStd] = IRFs_ridgeVAR_mod_asy(Yt, p, lambda, s, H);
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = Std;
        irfobj.sIRF = sIRF;
        irfobj.sStd = sStd;
        
    % ___________________________________________________
    %   Method: Bayesian VAR - Banbura, Giannone & Richielin (2010)
    case 'bvar'
        p = varargin{1};
        alpha = varargin{2};
        lambda = 1;
        epsilon = 1e-3;
        if length(varargin) > 2
            lambda = varargin{3};
        end
        if length(varargin) > 3
            epsilon = varargin{4};
        end
        % Bayesian VAR
        [IRF, bIRFdraws, ~, sIRF, sbIRFdraws] = ...
                                IRFs_BVAR(Yt, p, lambda, epsilon, H);
        % CIs
        phi = (1-alpha)/2;
        CIs  = quantile(bIRFdraws, [phi, 1-phi], 1);
        sCIs = quantile(sbIRFdraws, [phi, 1-phi], 1);
                   
        % Store
        irfobj.IRF  = IRF;
        irfobj.CIs  = CIs;
        irfobj.sIRF = sIRF;
        irfobj.sCIs = sCIs;
        
    % ___________________________________________________
    %   Method: Bayesian VAR - Banbura, Giannone & Richielin (2010)
    %           w/ Cross-Validation
    case 'bvar_cv'
        p = varargin{1};
        alpha = varargin{2};
        epsilon = 1e-3;
        nfolds = 3;
        cv_type = 'block_nondep';
        if length(varargin) == 3
            epsilon = varargin{3};
        end
        if length(varargin) > 3
            nfolds = varargin{4};
        end
        if length(varargin) > 4
            cv_type = varargin{5};
        end
        % Cross-validation
        [lambda, ~] = lambda_CV_BVAR(Yt, p, epsilon, nfolds, cv_type);
        % Bayesian VAR
        [IRF, bIRFdraws, ~, sIRF, sbIRFdraws] = ...
                                IRFs_BVAR(Yt, p, lambda, epsilon, H);
        % CIs
        phi = (1-alpha)/2;
        CIs  = quantile(bIRFdraws, [phi, 1-phi], 1);
        sCIs = quantile(sbIRFdraws, [phi, 1-phi], 1);
                   
        % Store
        irfobj.IRF  = IRF;
        irfobj.CIs  = CIs;
        irfobj.sIRF = sIRF;
        irfobj.sCIs = sCIs;
        
    % ___________________________________________________
    %   Method: Bayesian VAR - Giannone, Lenza & Primiceri (2015)
    case 'bvar_glp'
        p = varargin{1};
        alpha = varargin{2};
        mcmc = true;
        if length(varargin) == 3
            mcmc = varargin{3};
        end
        
        K = size(Yt, 1);
        phi = (1-alpha)/2;
        
        % Bayesian estimation
        % NOTE: using 'evalc()' suppresses the large output produced by
        %       the 'bvarGLP()' function
        if mcmc
            [~, res] = evalc("bvarGLP(Yt', p, 'mcmc', 1, 'MCMCconst', 1.6)");
        else
            [~, res] = evalc("bvarGLP(Yt', p, 'hz', H)");
        end
        % res = bvarGLP(Yt', p, 'mcmc', 1, 'MCMCconst', 1.6);
        
        % IRFs at the posterior mode
        beta = res.postmax.betahat;
        sigma = res.postmax.sigmahat;
        
        IRF  = zeros(K^2, H+1);
        sIRF = zeros(K^2, H+1);
        CIs  = zeros(2, K^2, H+1);
        sCIs = zeros(2, K^2, H+1);
        
        % Go through all shocks
        for k = 1:K
            [IRF_j, sIRF_j] = bvarIrfs_mod(beta, sigma, k, H+1);
            
            IRF(1+(k-1)*K:k*K,:)  = IRF_j';
            sIRF(1+(k-1)*K:k*K,:) = sIRF_j';
            
            if mcmc         % estimate CIs from MCMC draws
                ndraws = size(res.mcmc.beta, 3);
                
                % IRFs at each draw
                D_IRF_j  = zeros(H+1, K, ndraws);
                D_sIRF_j = zeros(H+1, K, ndraws);
                for jg = 1:ndraws
                    beta  = res.mcmc.beta(:,:,jg);
                    sigma = res.mcmc.sigma(:,:,jg);
                    % Use the modified 'bvarIrfs_mod()' function to
                    % compute in one shot both reduced form and
                    % structural IRFs
                    [dirfj, dsirfj] = bvarIrfs_mod(beta, sigma, k, H+1);
                    
                    D_IRF_j(:,:,jg)  = dirfj;
                    D_sIRF_j(:,:,jg) = dsirfj;
                end
                
                % CIs
                qCI = quantile(D_IRF_j, [phi, 1-phi], 3);
                CIs(1,1+(k-1)*K:k*K,:) = qCI(:,:,1)';
                CIs(2,1+(k-1)*K:k*K,:) = qCI(:,:,2)';
                qsCI = quantile(D_sIRF_j, [phi, 1-phi], 3);
                sCIs(1,1+(k-1)*K:k*K,:) = qsCI(:,:,1)';
                sCIs(2,1+(k-1)*K:k*K,:) = qsCI(:,:,2)';
            end
        end
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.CIs  = CIs;
        irfobj.sIRF = sIRF;
        irfobj.sCIs = sCIs;
        
    % ___________________________________________________
    %   Method: Stein Combination Shrinkage - Hansen (2016)
    %   
    % NOTE: this method only return pointwise structural IRFs, as
    % originally coded in Hansen (2016), therefore it should NOT be used to
    % study CIs with associated statistics, but only MSE/Bias.
    case 'comb_stein_hansen'
        p = varargin{1};
        
        K = size(Yt, 1);
        
        IRF = nan(K^2, H+1);

        [sIRF, ~] = cvar_ir(Yt', p, H);
        sIRF = reshape(sIRF, H+1, [])';
        
        % Store
        irfobj.IRF  = IRF;
        irfobj.Std  = nan(K^2, H+1);
        irfobj.sIRF = sIRF;
        irfobj.sStd = nan(K^2, H+1);
        
    otherwise
        error("Unrecognized IRF method")
end

end

% #####