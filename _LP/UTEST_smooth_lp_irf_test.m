%  SMOOTH LP IRF TEST
%   Giovanni Ballarin, August 2021
%

clear

seed = 198233433;
rng(seed)

% FLAG
SIMULATED = false;

% Length of IRFs
H_min = 0;
H_irf = 20;

%% Data
if SIMULATED
    % Construct VARMA model
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
    Sigma_u = P * P';
    
    K = 3;
    Yt = simVARMA_p_1(A1, M1, Sigma_u, 500);
    
    % True IRFs
    [IRF, sIRF] = varmaIRF(A1, M1, P, H_irf);
else
    load('data')
    
    Yt = table2array(data(:,2:end))';
end

%% IRF Estimation
% LP
[lpIRF,lpStd,s_lpIRF,s_lpStd] = IRFs_LP(Yt, 3, H_irf);
s_lpConf = zeros(2, 9, H_irf+1);
s_lpConf(1,:,:) = s_lpIRF + s_lpStd*norminv(0.95);
s_lpConf(2,:,:) = s_lpIRF + s_lpStd*norminv(0.05);

% Smooth LP
lambda = 0.01;

tic
[slpIRF,slpStd] = IRFs_SmoothLP(Yt, 4, 3, lambda, H_min:H_irf, [1,3], [1,2]);
toc
sslpConf = zeros(2, H_irf+1);
sslpConf(1, :) = slpIRF + slpStd*norminv(0.95);
sslpConf(2, :) = slpIRF + slpStd*norminv(0.05);

% ORIGINAL CODE
y  = Yt(1,:)'; % endogenous variable
x  = Yt(3,:)'; % endoegnous variable related to the shock 
w  = [ Yt([1, 2],:)' , lagmatrix( Yt' , 1:4 ) ]; % control variables
w( ~isfinite(w) ) = 0;

tic
slp = locproj(y, x, w, H_min, H_irf, 'smooth', 3, lambda); %IR from Smooth Local Projection
slp = locproj_conf(slp, H_irf, lambda/2);
toc

tic
slp_bb2019 = locproj_latest(y, x, w, H_min, H_irf, 'smooth', 3, lambda);
toc
%% Plot

N = 7;

figure
hold on
% Confidence bands
plot(polyshape([0:H_irf, H_irf:-1:0]', ...
               [squeeze(s_lpConf(1,N,:)); flip(squeeze(s_lpConf(2,N,:)))]), ...
         'FaceAlpha', .2, 'FaceColor', [.5,.5,.5], 'EdgeColor', 'none');
plot(polyshape([0:H_irf, H_irf:-1:0]', ...
               [sslpConf(1,:)'; flip(sslpConf(2,:))']), ...
         'FaceAlpha', .2, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
plot(polyshape([H_min:H_irf, H_irf:-1:H_min]', ...
               [slp.conf(H_min+1:end,1); flip(slp.conf(H_min+1:end,2))]), ...
         'FaceAlpha', .2, 'FaceColor', '#D95319', 'EdgeColor', 'none');
% Pointwise IRF
if SIMULATED
    p0 = plot(0:H_irf, sIRF(N,:), 'k');
else
    p0 = plot(0:H_irf, zeros(1,H_irf+1), 'k:');
end
p1 = plot(0:H_irf, slpIRF(:), 'Color', '#0072BD', 'Marker', '.');
p2 = plot(0:H_irf, slp.IR, 'Color', '#D95319', 'Marker', '.');
p3 = plot(0:H_irf, s_lpIRF(N,:), '--', 'Color', '#A2142F', 'Marker', '.');
hold off
grid 
xlim([0 H_irf])
yline(0, '--', 'Color', [.5,.5,.5])
legend([p0, p1, p2, p3], "True", "New SLP", "SLP BB2019", "LP")

% #####