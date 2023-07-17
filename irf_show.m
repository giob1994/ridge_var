% irf_show()
%   Show a realization of the chosen IRF estimation methods
%   Giovanni Ballarin, Aug 2021
%

function [plotobj] = irf_show(IRF_id, options, methods, colors)
% Inputs:
%
%   IRF_id      index of the IRF to show the simulations of
%
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
%       - CIs      show CIs in plot
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
%   colors      array, indexes of colors to use
%

B = options.B;
T = options.T;
H = options.H;
type = options.type;

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
    res(n).IRF    = zeros(K^2, H+1);
    res(n).CIs    = zeros(2, K^2, H+1);
end

% Simulate model
Yt = options.model(T);
Yt = Yt - mean(Yt, 2);

tic
% Construct IRFs and CIs
for n = 1:length(methods)
    if ~methods{n}.int      % method returns variances
        irfobj = methods{n}.fun(Yt, H);
        if strcmp(options.type, 'reduced')
            res(n).IRF = irfobj.IRF;
            Std_est = irfobj.Std;
        elseif strcmp(options.type, 'structural')
            res(n).IRF = irfobj.sIRF;
            Std_est = irfobj.sStd;
        end
        
        res(n).CIs(1,:,:) = res(n).IRF - qnorm * Std_est;
        res(n).CIs(2,:,:) = res(n).IRF + qnorm * Std_est;
        
    else                    % method returns CIs
        irfobj = methods{n}.fun(Yt, H, alpha);
        if strcmp(options.type, 'reduced')
            res(n).IRF = irfobj.IRF;
            CIs_est = irfobj.CIs;
        elseif strcmp(options.type, 'structural')
            res(n).IRF = irfobj.sIRF;
            CIs_est = irfobj.sCIs;
        end
        
        res(n).CIs(1,:,:) = CIs_est(1,:,:);
        res(n).CIs(2,:,:) = CIs_est(2,:,:);
        
    end
    fprintf('.')
end
fprintf('\n')
time = toc;

%% Plot

if isfield(options, 'CIs')
    flag_CIs = options.CIs;
else
    flag_CIs = true;
end

fig = figure();
tiledlayout(1, 1, 'Padding', 'compact', 'TileSpacing', 'compact'); 

sgt = sgtitle(...
    sprintf("Type: '%s', Params: T = %d, B = %d, H = %d", ...
            type, T, B, H) ...
);
sgt.FontSize = 9;

nexttile
%
plot(0:H, IRF(IRF_id,:), '--k', 'LineWidth', 0.8);
%
hold on
ps = [];
for n = 1:size(res, 1)
    if ~isempty(colors)
        cn = defaultPlotColors(colors(n));
    else
        cn = defaultPlotColors(n);
    end
    % CIs
    if flag_CIs
        cin_l = squeeze(res(n).CIs(1,IRF_id,:));
        cin_u = squeeze(res(n).CIs(2,IRF_id,:));
        plot(polyshape([0:H, H:-1:0]', [cin_l; flip(cin_u)]), ...
                'FaceAlpha', .1, 'FaceColor', cn, 'EdgeColor', 'none');
    end
    % IRF
    ps(n) = plot(0:H, res(n).IRF(IRF_id,:), ...
                        'Color', cn, ...
                        'LineWidth', 1);
end
%
% yl = ylim;
xlim([0, H])
ylim auto
xticks(0:4:H)
ylabel(sprintf('IRF %d', IRF_id))
set(gca, 'FontSize', 12)
grid

lg = legend(ps, {res.method}, 'Location', 'southoutside', 'Orientation', 'horizontal');
lg.Layout.Tile = 'south';


end

% #####