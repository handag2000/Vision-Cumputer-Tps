close all, clear all, clc
%%
X=imread('cameraman.tif');
figure(1), imshow(X), title('image_imput'); %affiché l'image originale.
Row = 164; % ligne numéro 164 de l'image
x = double(X(Row,:));% convertir vers le double toute la ligne numéro 164.
Col = size(X,2);
figure(2), subplot(3,1,1), plot(1:Col,x,'k', 'LineWidth',2)
xlabel('Pixel number'), ylabel('Amplitude') % Les axes x et y sont étiquetés avec les noms "Pixel number" et "Amplitude" respectivement.
x1=diff(x);  % dérivé premier de x.
x2=diff(x1); % dérivé seconde de x1.

subplot(3,1,2),plot(1:Col-1,x1,'g','LineWidth',2); %affiché les dérivés premier
xlabel('Pixel number'), ylabel('Amplitude')
subplot(3,1,3),plot(1:Col-2,x2,'r','LineWidth',2); %affiché le dérivé second.
xlabel('Pixel number'), ylabel('Amplitude')

%on peu t faire ça sur 2lignes ou images entiers.
%% Application sur une image 
[dx dy]=gradient(double(X));

%calcule de second dérivé 
d2=del2(double(X));

%
dx=normalize(dx);
dy=normalize(dy);
d2=normalize(d2);
% affiché les dérivés images.

figure;
subplot(141), imshow(X), title('original image');
subplot(142), imshow(uint8(dx)), title('dérivé premier dx.');

subplot(143), imshow(uint8(dy)), title('dérivé premier dy.');
subplot(144), imshow(uint8(abs(dx)+abs(dy))), title('la somme normalisé');



%%
[dx dy]=gradient(double(X)); % Calcul du gradient de l'image X par rapport aux coordonnées x et y.
magnitude=sqrt(dx.^2+dy.^2); %Calcul de la magnitude du gradient.
direction= atan2(dy,dx); %Calcul de la direction du gradient.
%calcule de second dérivé 
d2=del2(double(X));

%

% affiché les dérivés images.

figure;
subplot(131), imshow(X), title('original image');
subplot(132), imshow(magnitude,[]), title('Gradient magnitude');

subplot(133), imshow(direction), title('Gradient direction');
colormap('hsv'); %définit la colormap à utiliser pour l'affichage de l'image de direction du gradient.
colorbar; %ajoute une barre de couleur à côté de l'image de direction du gradient

%%
I=imread('circuit.tif');
%Pour les filtres : Roberts, Prewitt et Sobel.
F_R=edge(I,'Roberts');
F_P=edge(I,'Prewitt');
F_S=edge(I,'Sobel');
% Affiche des résultats.
figure, 
subplot(1,4,1), imshow(I);title('img original');
subplot(1,4,2), imshow(F_R);title('filtre de Roberts');
subplot(1,4,3), imshow(F_P);title('filtre de Prewitt');
subplot(1,4,4), imshow(F_S);title('filtre de Sobel');
%%
I=imread('peppers.png');
I=rgb2gray(I);
F_C=edge(I,'canny');
F_LOG=edge(I,'Log');

figure, 
subplot(1,3,1), imshow(I);title('img original');
subplot(1,3,2), imshow(F_C);title('filtre de Canny');
subplot(1,3,3), imshow(F_LOG);title('filtre de Log');


%%
I=imread('peppers.png'); % lire l'image
I=rgb2gray(I);           % convertir de rgb vers gray
h=fspecial('gaussian',[15 15],6);  % création de filtre.
I1=imfilter(I,h);                   % application de filtre h
h2=fspecial('gaussian',[30 30],12); %création de filtre.
I2=imfilter(I,h2);  % application de filtre h2

%application de filtre DE détection de conteurs.
I1_C=edge(I1,'canny');
I1_L=edge(I1,'Log');
I2_C=edge(I2,'canny');
I2_L=edge(I2,'Log');

% affichage des resultats.

figure,
subplot(2,3,1), imshow(I);title('img original');
subplot(2,3,2), imshow(I1_C);title('I filtré det Canny');

subplot(2,3,3), imshow(I1_L);title('I filtré det Log');
subplot(2,3,4), imshow(I2_C);title('I filtré det Canny');
subplot(2,3,5:6), imshow(I2_L);title('I filtré det Log');









