function mD=GenDuplicationMatrix(k)
            
mD=zeros(k*k,k*(k+1)/2);
s=(1:1:k)';
var=zeros(k,k);
var(:,1)=s;
x=k-1;
for i=2:k;
   var(i:k,i)=s(i:k,1)+x;
   var(i-1,i:k)=var(i:k,i-1)';
   x=x+k-i;
end;
var=reshape(var,k^2,1);
for i=1:k^2;
   mD(i,var(i,1))=1;
end;

