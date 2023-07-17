% BAYESIAN VAR IRF TEST
%   Giovanni Ballarin, July 2021
%

clear

seed = 198233433;
rng(seed)

% Length of IRFs
H_irf = 50;

%% Construct VARMA model

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

K = size(A1, 1);

%% True IRFs

[IRF, sIRF] = varmaIRF(A1, M1, P, H_irf);

%%

Yt = simVAR_p(A1, Sigma_u, 500);
% Yt = Yt - mean(Yt, 2);

sigma_ = diag(Sigma_u);
lambda_ = 1;
epsilon_ = 0.01;

[B, Sig_U, Sig_B] = BVAR(Yt, 2, sigma_, lambda_, epsilon_);
B = B';

[bIRF, bIRFdraws, P, sbIRF, sbIRFdraws, ~, ~, ~] = ...
                       IRFs_BVAR(Yt, 2, sigma_, lambda_, epsilon_, H_irf);
                   
% CIs
bIRF_q9 = quantile(bIRFdraws, [.05, .95], 1);
bIRF_q6 = quantile(bIRFdraws, [.11, .89], 1);

sbIRF_q9 = quantile(sbIRFdraws, [.05, .95], 1);
sbIRF_q6 = quantile(sbIRFdraws, [.11, .89], 1);
                   
%% Plot

N = 1;

figure
hold on
% Confidence bands
plot(polyshape([0:H_irf, H_irf:-1:0]', ...
               [squeeze(sbIRF_q9(1,N,:)); flip(squeeze(sbIRF_q9(2,N,:)))]), ...
         'FaceAlpha', .2, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
plot(polyshape([0:H_irf, H_irf:-1:0]', ...
               [squeeze(sbIRF_q6(1,N,:)); flip(squeeze(sbIRF_q6(2,N,:)))]), ...
         'FaceAlpha', .2, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
% Pointwise IRF
plot(0:H_irf, sIRF(N,:), 'k', 'Marker', '.')
plot(0:H_irf, sbIRF(N,:), 'b', 'Marker', '.')
hold off
grid 
yline(0, '--', 'Color', [.5,.5,.5])

% #####