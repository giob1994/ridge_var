function mK=GenCommutationMatrix(k)

	mK = zeros(k^2,k);
	% first n columns
	r=1;
    for c=1:k;
	   mK(r,c)=1;
	   r=r+k; 
    end;       	   	

   	zold= mK;
   	% rest of colums  
    for cc=1:k-1;
        mK = [mK,[zeros(cc,k);zold(1:k^2-cc,:)]];    
    end; 	


