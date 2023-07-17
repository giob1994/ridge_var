function [result, eigens] = stationary(B)
%STATIONARY Check if a VAR parameter specification is stable or not
%

K  = size(B, 1);
Kp = size(B, 2);

Bc = [B; [eye(Kp-K), zeros(Kp-K, K)]];
    
eigens = sort(abs(eig(Bc)),'desc');
result = max(eigens) < 1;

end

