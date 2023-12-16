%%
A=imread('cameraman.tif'); imshow(A)   % lire l'image cameraman.tif 
B=imread('rice.png'); imshow(B)        % lire l'image ricr.png
C=A+B ; imshow(C)                      % faire l'addition entre les deux image A et B est l'affiché . 
D=imadd(A,B); imshow(D)                % faire l'add à partir de la l'instruction imadd(). 

figure(1)                              % afficher les deux image en meme figure pour comparer.
subplot(1,2,1), imshow(C)
subplot(1,2,2), imshow(D)
%%
K=imadd(A,B,'uint16');      
figure(2), imshow(K,[])  % on utilise les crochés car l'instruction imshow(X) est difini seulement pour les uint8 pour dépassé 255 il faut ajouter []
%%
E=A+100; imshow(E)   % on ajoute à l'image A une constante 
%%
I=imread('moon.tif');       % lire l'image moon exister dans le toolbox 
J=immultiply(I,0.5);        % faire mla multiplication de l'image I * 0.5
subplot(1,2,1), imshow(I)   % aficher I 
subplot(1,2,2), imshow(J)    % aficher J 
%%
I=imread('rice.png');  % lire l'image rice.png
J1 =imresize(I, 0.5);  % redimensioner I par un scale de 0.5 .
J2 =imresize(I, 2);    % redimensioner I par un scale de 2 .
J3 =imresize(I, 4);    % redimensioner I par un scale de 4 .

figure (4),            % afficher toutes les images en meme figure 
subplot(2,2,1), imshow(I)
subplot(2,2,2), imshow(J1)
subplot(2,2,3), imshow(J2)
subplot(2,2,4), imshow(J3)
whos
%%  
I1 = imread('cameraman.tif');
I2 = grayslice(I1,128); figure(1), imshow(I2,gray(128));
% la commande grayslice() permet de convertir une image en niveaux de gris en image indexée à l'aide d'un seuillage à plusieurs niveaux

I3 = grayslice(I1,64);  figure(2), imshow(I3,gray(64)); % les entre 0 et 64
I4 = grayslice(I1,32); figure(3), imshow(I4,gray(32));
I5 = grayslice(I1,16); figure(4), imshow(I5,gray(16));
I6 = grayslice(I1,8); figure(5), imshow(I6,gray(8));
I7 = grayslice(I1,4); figure(6), imshow(I7,gray(4));     
I8 = grayslice(I1,2); figure(7), imshow(I8,gray(2));     % img binaire les valeurs 0 et 1

%%
A= imread('lena.jpg');
B=imrotate(A,45); imshow(B)  % faire la rotation de l'image A avec l'angle 45 
cb = checkerboard; imshow(cb)% permet de créer une image en damier 
xform = [ 1 0 0 ; 0 1 0; 40 40 1];   
tform_translate=maketform('affine',xform);      % Faire et appliquer une transformation affine.          
[cb_trans xdata ydata]= imtransform(cb, tform_translate); %recupérer data de l'image transformer
figure(5), imshow(cb_trans)                   % afficher l'image 
%%
img=imread('cercle1.png');       % lire l'image couleur Cercle1.png
% les trois commandes suivantes pour sépareé le trois couches Rouge, vert
% et bleu.
R=img(:,:,1); imshow(R)          
G=img(:,:,2); imshow(G)
B=img(:,:,3); imshow(B)
% faire une translation pour modifier la position de cercle
R1=imtranslate(R,[-1 -20]);
G1=imtranslate(G,[30 12]);
B1=imtranslate(B,[-30 21]);
% creé une image s'appele img1 contient les trois couches modifier .
img1(:,:,1)=R1; 
img1(:,:,2)=G1;
img1(:,:,3)=B1;

figure(1), imshow(img1)  % afficher l'image final 




