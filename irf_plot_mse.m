% irf_plot_mse()
%   Plot coverage properties estimated IRF functions
%   Giovanni Ballarin, Aug 2021
%

function [objplot] = irf_plot_mse(objsim, IRF_id, options, save_flag)
% Inputs:
%   objsim      an output object of the 'irf_simulate()' function
%
%   IRF_id      index of the IRF to show the simulations of
%   
%   options     struct:
%       - title         set the title of the plot
%       - lims          set the limits of the plot
%                       struct:
%                           - c_ylim
%                           - c_xlim
%                           - l_ylim
%                           - l_xlim
%
%   save_flag   save the plots to file or not
%

T = objsim.T;
B = objsim.B;
H = objsim.H;
alpha = objsim.alpha;
type = objsim.type;

% Extract results struct
results = objsim.simres;

%% Plot

fig = figure();
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact'); 

sgt = sgtitle(...
    sprintf("Type: '%s', Params: T = %d, B = %d, H = %d", ...
            type, T, B, H) ...
);
sgt.FontSize = 9;

% Coverage plot
nexttile
%
hold on
ps = [];
for n = 1:size(results, 1)
    if strcmp(type, 'reduced')
        ps(n) = plot(1:H, results(n).MSE(IRF_id,2:end), ...
                        'Color', defaultPlotColors(n), ...
                        'Marker', '.', 'LineWidth', 1);
    elseif strcmp(type, 'structural')
        ps(n) = plot(0:H, results(n).MSE(IRF_id,:), ...
                        'Color', defaultPlotColors(n), ...
                        'Marker', '.', 'LineWidth', 1);
    end
end
%
% yl = ylim;
% ylim([min(yl(1), 0.75), 1])
xlim([0, H])
xticks(0:4:H)
ylabel('MSE')
set(gca, 'FontSize', 12)
grid

% Length plot
nexttile
%
hold on
for n = 1:size(results, 1)
    if strcmp(type, 'reduced')
        plot(1:H, results(n).Bias(IRF_id,2:end), ...
                'Color', defaultPlotColors(n), ...
                'Marker', '.', 'LineWidth', 1);
    elseif strcmp(type, 'structural')
        plot(0:H, results(n).Bias(IRF_id,:), ...
                'Color', defaultPlotColors(n), ...
                'Marker', '.', 'LineWidth', 1);
    end
end
%
xlim([0, H])
xticks(0:4:H)
ylabel('Bias')
set(gca, 'FontSize', 12)
grid

lg = legend(ps, {results.method}, 'Location', 'southoutside', 'Orientation', 'horizontal');
lg.Layout.Tile = 'south';

end

% #####