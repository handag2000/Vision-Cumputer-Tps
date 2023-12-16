%%
RGB=imread('lena_color.bmp'); % lire une image color 
R=RGB(:,:,1); G=RGB(:,:,2); B=RGB(:,:,3);   % séparé les couches 
%%
HSV=rgb2hsv(Img);               % convertir de RGB to HSV
H=HSV(:,:,1); S=HSV(:,:,2); V=HSV(:,:,3);
%%
figure(1)
subplot(4,2,1), imshow(RGB)
subplot(4,2,2), imshow(HSV)
subplot(4,2,3), imshow(R)
subplot(4,2,4), imshow(H)
subplot(4,2,5), imshow(G)
subplot(4,2,6), imshow(S)
subplot(4,2,7), imshow(B)
subplot(4,2,8), imshow(V)

%%
YUV = rgb2ycbcr(RGB);   % convertir de rgb vers YUV
Y=YUV(:,:,1); U=YUV(:,:,2); V=YUV(:,:,3);

figure(2)
subplot(1,4,1); imshow(YUV);
subplot(1,4,2); imshow(Y);
subplot(1,4,3); imshow(U);
subplot(1,4,4); imshow(V);

%%
RGB1=ycbcr2rgb(YUV);
figure(3);
subplot(1,2,1), imshow(RGB);
subplot(1,2,2), imshow(RGB1);

%%
GRIS=rgb2gray(RGB);
figure(4);
subplot(1,2,1), imshow(RGB);
subplot(1,2,2), imshow(GRIS);
