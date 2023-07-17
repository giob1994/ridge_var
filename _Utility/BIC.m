function [p, BICmin] = BIC(Yt, maxp)
%BIC Selected VAR order via Bayesian IC 
%   

T = size(Yt, 2);
K = size(Yt, 1);

obj = zeros(1,maxp);
for m = 1:maxp
    % Estimate VAR(m) model
    [~, Sig_U_m, ~] = LS_VAR(Yt, m);
    obj(m) = log(det(Sig_U_m)) + log(T)*m*K^2/T;
end
% Find minimizer
[BICmin, p] = min(obj);

end

