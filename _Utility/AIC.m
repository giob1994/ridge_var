function [p, AICmin] = AIC(Yt, maxp)
%AIC Selected VAR order via Akaike IC 
%   

T = size(Yt, 2);
K = size(Yt, 1);

obj = zeros(1,maxp);
for m = 1:maxp
    % Estimate VAR(m) model
    [~, Sig_U_m, ~] = LS_VAR(Yt, m);
    obj(m) = log(det(Sig_U_m)) + 2*m*K^2/T;
end
% Find minimizer
[AICmin, p] = min(obj);

end

