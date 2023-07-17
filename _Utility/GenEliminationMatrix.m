function mL=GenEliminationMatrix(k)
            
k1=1;  
k2=k;  
k3=1;  
k4=k;      
mL=zeros(k*(k+1)/2,k*k);
for i = 0:k-1;
   mL(k1:k2,k3:k4)=eye(k-i);
   k1=k2+1; 
   k2=k2+k-i-1; 
   k3=k4+1+i+1; 
   k4=k4+k;
end;
