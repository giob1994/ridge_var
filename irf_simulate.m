% irf_simulate()
%   Simulate a process and compute IRF functions with associated stats
%   Giovanni Ballarin, Aug 2021
%

function [objsim] = irf_simulate(options, methods)
% Inputs:
%   options     struct:
%       - model(T) function, returns a draw from a DGP model
%       - seed     rng replication seed
%       - B        number of replications
%       - T        sample size of the model draws
%       - H        IRF horizon
%       - alpha    confidence level of CIs
%       - type     type if IRF to compute (reduced / structural)
%       - IRF(H)   function, returns ground truth IRFs as
%                              [IRF, sIRF]
%   
%   methods     cell{}:
%       - struct:
%           - desc                  desciption of the IRF inference method
%           - int                   boolean, the method already returns CIs 
%                                   for IRFs (e.g. Bayesian CIs)
%           - fun(Y, H, (alpha))    function to call on data 'Y' to get IRFs.
%                                   If 'int == false', 'fun' must return an 
%                                   object with fields
%                                       .IRF / .Std / .sIRF / .sStd
%                                   If 'int == true', 'fun' must also accept 
%                                   a 3rd argument 'alpha' for the CI level
%                                   and must return an object with fields:
%                                       .IRF / .CIs / .sIRF / .sCIs
%

B = options.B;
T = options.T;
H = options.H;

alpha = options.alpha;
qnorm = norminv(1-((1-alpha)/2), 0, 1);

rng(options.seed);
% parseeds = seed + randperm(B);

% True impulse responses
[IRF_true, sIRF_true] = options.IRF(H);
if strcmp(options.type, 'reduced')
    IRF = IRF_true;
elseif strcmp(options.type, 'structural')
    IRF = sIRF_true;
end

K = sqrt(size(IRF, 1)); % size of the model

%% Simulation

if length(methods) < 1
    error("No methods selected to compute IRFs")
end

% Create storage object
res = repmat(struct('method', []), length(methods), 1);
for n = 1:length(methods)
    res(n).method = methods{n}.desc;
    res(n).type   = options.type;
    res(n).Cover  = zeros(K^2, H+1);
    res(n).Length = zeros(K^2, H+1);
    res(n).MSE    = zeros(K^2, H+1);
    res(n).Bias   = zeros(K^2, H+1);
end

tic 
for b = 1:B
% parfor b = 1:B
    % Simulate model
    Yt = options.model(T);
    Yt = Yt - mean(Yt, 2);

    % Construct IRFs and CIs
    for n = 1:length(methods)
        fprintf('.')
        
        if ~methods{n}.int      % method returns asymptotic variances
            irfobj = methods{n}.fun(Yt, H);
            if strcmp(options.type, 'reduced')
                IRF_est = irfobj.IRF;
                Std_est = irfobj.Std;
            elseif strcmp(options.type, 'structural')
                IRF_est = irfobj.sIRF;
                Std_est = irfobj.sStd;
            end
            
            Covg_b = zeros(K^2, H+1);
            for k = 1:K^2
                for h = 1:H+1
                    if (IRF(k,h) >= IRF_est(k,h)-qnorm*Std_est(k,h) && ...
                            IRF(k,h) <= IRF_est(k,h)+qnorm*Std_est(k,h))
                        Covg_b(k,h) = 1;
                    end
                end
            end
            res(n).Cover  = res(n).Cover + Covg_b/B;
            res(n).Length = res(n).Length + (2*qnorm*Std_est)/B;
            res(n).MSE    = res(n).MSE + (IRF - IRF_est).^2/B;
            res(n).Bias   = res(n).Bias + (IRF - IRF_est)/B;
            
        else                    % method returns CIs
            irfobj = methods{n}.fun(Yt, H, alpha);
            if strcmp(options.type, 'reduced')
                IRF_est = irfobj.IRF;
                CIs_est = irfobj.CIs;
            elseif strcmp(options.type, 'structural')
                IRF_est = irfobj.sIRF;
                CIs_est = irfobj.sCIs;
            end
            
            Covg_b = zeros(K^2, H+1);
            for k = 1:K^2
                for h = 1:H+1
                    % disp(IRF(k,h))
                    % disp(CIs_est(1,k,h))
                    % disp(CIs_est(2,k,h))
                    if (IRF(k,h) >= CIs_est(1,k,h) && ...
                            IRF(k,h) <= CIs_est(2,k,h))
                        Covg_b(k,h) = 1;
                    end
                end
            end
            res(n).Cover  = res(n).Cover + Covg_b/B;
            res(n).Length = res(n).Length + squeeze(CIs_est(2,:,:) - ...
                                                        CIs_est(1,:,:))/B;
            res(n).MSE    = res(n).MSE + (IRF - IRF_est).^2/B;
            res(n).Bias   = res(n).Bias + (IRF - IRF_est)/B;
            
        end
    end
    fprintf('\n')
end
time = toc;

objsim         = options;
objsim.IRF     = IRF;
objsim.simres  = res;
objsim.time    = time;


end

% #####