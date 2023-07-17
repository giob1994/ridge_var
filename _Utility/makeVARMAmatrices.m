function [A1,M1] = makeVARMAmatrices(K)
%makeVARMAmatrices

rng(90214)

% VARMA matrices from Kilian & Kim (2011)
A1_k = [  0.5417 -0.1971 -0.9395;
    0.04    0.9677  0.0323;
    -0.0015  0.0829  0.8080  ];
M1_k = [ -0.1428 -1.5133 -0.7053;
    -0.0202  0.0309  0.1561;
    0.0227  0.1178 -0.0153  ];

if not(exist('Rmatrix_VARMA.mat','file'))
    disp('!!! Simulation parameters do not exists');
else
    switch K
        case 3
            % K = 3 Kilian & Kim (2011) model
            A1 = A1_k;
            M1 = M1_k;
        case 8
            load('Rmatrix_VARMA.mat','Phi_8','The_8')
            % K = 8 simulated model
            A1 = Phi_8;
            M1 = The_8;
        case 11
            load('Rmatrix_VARMA.mat','Phi_8','The_8')
            % K = 11 augmented Kilian & Kim (2011) model
            A1 = zeros(11); % 0.01*randn(11,11);
            A1(1:3,1:3) = A1_k;
            A1(4:end,4:end) = Phi_8;
            M1 = zeros(11);
            M1(1:3,1:3) = M1_k;
            M1(4:end,4:end) = The_8;
        case 15
            load('Rmatrix_VARMA.mat','Phi_15','The_15')
            % K = 15 simulated model
            A1 = Phi_15;
            M1 = The_15;
        case 18
            load('Rmatrix_VARMA.mat','Phi_15','The_15')
            % K = 18 augmented Kilian & Kim (2011) model
            A1 = zeros(18); % 0.01*randn(18,18);
            A1(1:3,1:3) = A1_k;
            A1(4:end,4:end) = Phi_15;
            M1 = zeros(18);
            M1(1:3,1:3) = M1_k;
            M1(4:end,4:end) = The_15;
        otherwise
            disp("! Model not defined !")
    end
end

end

