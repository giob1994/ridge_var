function obj = locproj_latest(varargin)
% locproj
%
%   locporj(y,x,w,h1,H,type)
%   locporj(y,x,w,h1,H,type,r,lambda)
%
%   
    switch length(varargin)
        case 6
            y    = varargin{1};
            x    = varargin{2};
            w    = varargin{3};
            h1   = varargin{4};
            H    = varargin{5};
            type = varargin{6};

        case 8
            y    = varargin{1};
            x    = varargin{2};
            w    = varargin{3};
            h1   = varargin{4};
            H    = varargin{5};
            type = varargin{6};            
            r    = varargin{7};
            lam  = varargin{8};
            
        otherwise
            error('wrong number of input arguments')
        
    end    

    obj = struct();
      
    std = strcmp('reg',type);
    T  = length(y);
    HR = H + 1 - h1;
    
    % constructr the B-spline basis functions
    if ~std
        B = bspline( (h1:H)' , h1 , H+1 , H+1-h1 , 3 );
        K = size( B , 2 );
    else
        K = HR;
    end

    % building up the regression representation of the local projection
    idx = nan( (H+1)*T , 2 );
    Y   = nan( (H+1)*T , 1 );
    Xb  = zeros( (H+1)*T , K );
    Xc  = zeros( (H+1)*T , HR , size(w,2)+1 );
    
    w = [ ones(T,1) w ];
    
    for t = 1:T-h1
        
        idx_beg = (t-1)*HR + 1;
        idx_end = t*HR;

        idx( idx_beg:idx_end , 1 ) = t;
        idx( idx_beg:idx_end , 2 ) = h1:H;
        
        % y
        y_range = (t+h1) : min((t+H),T)';
        Y( idx_beg:idx_end ) = [ y( y_range ) ; nan(HR-length(y_range),1) ];

        % x
        if std
            Xb( idx_beg:idx_end , : ) = eye(HR)*x(t);
        else
            Xb( idx_beg:idx_end , : ) = B*x(t);
        end

        % w
        for i = 1:size(w,2)
            Xc( idx_beg:idx_end , : , i ) = eye(HR)*w(t,i);
        end
        
    end
    
    X = Xb;
    for i = 1:size(w,2)
        X = [X Xc(:,:,i)];
    end
    
    select = isfinite(Y);  
    idx = idx(select,:);
    Y   = Y(select);
    X   = X(select,:);
    X   = sparse(X);

    % estimation
    IR  = zeros(HR+1,1);
    
    if std

        theta     = ( X'*X )\( X'*Y );
        IR((1+h1):(1+HR)) = theta(1:K);
    
    else

        P = zeros( size(X,2) );

        D = eye(K);
        for k = 1:r 
            D = diff(D);
        end
        
        P(1:K,1:K) = D' * D;

        theta = ( X'*X + lam*P )\( X'*Y );
        
        IR((1+h1):(1+HR)) = B * theta(1:K);
    end
    
    % pack everything up
    obj.T   = T;
    obj.HR  = HR;
    obj.K   = K;
    
    if ~std
        obj.B   = B;
    end
    
    obj.idx   = idx;
    obj.Y     = Y;
    obj.X     = X;
    obj.theta = theta;
    obj.IR    = IR;
    
    
    % debug stuff
    % Bs is for display / debugging pourposes only
    % Bs = bspline( (h1:0.1:H)' , h1 , H+1 , H+1-h1 , 3 );
    % obj.Bs = Bs;
end

function B = bspline(x, xl, xr, ndx, bdeg)
    dx = (xr - xl) / ndx;
    t = xl + dx * [-bdeg:ndx-1];
    T = (0 * x + 1) * t;
    X = x * (0 * t + 1);
    P = (X - T) / dx;
    B = (T <= X) & (X < (T + dx));
    r = [2:length(t) 1];
    for k = 1:bdeg
        B = (P .* B + (k + 1 - P) .* B(:, r)) / k;
    end
end
