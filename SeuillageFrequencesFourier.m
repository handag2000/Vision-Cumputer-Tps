function masque=SeuillageFrequencesFourier(X,R,param)
[M,N]=size(X);
masque=zeros(M);
x=1:M; y=1:M;
if mod(M,2)==0
c=M/2+1;
else
c=(M+1)/2;
end
[x_vect, y_vect]=ndgrid(x,y);
if param==0
masque((x_vect-c).^2 +(y_vect-c).^2 < R^2)=1;
else
masque((x_vect-c).^2 +(y_vect-c).^2 > R^2)=1;
end
end