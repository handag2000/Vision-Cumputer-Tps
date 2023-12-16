%% Part one 
clear all; close all; clc;
%%filtrage spatial
%%
I=imread('chambre.jpg'); %lire l'image existé dans le toolbox cameraman.tif.
 fltr=fspecial('averag');   % filtr moyenneur par defaut c'est de taille 3x3.
 Iav=imfilter(I,fltr);       %appliquer le filtre sur l'image I.
 
 % utilisation de subplot pour comparer les resultats.
 figure, subplot(1,2,1), imshow(I); title('l"image original');
  subplot(1,2,2), imshow(Iav);title('l"image filtrer'); 
 
%% creation d'un filtre moyenneur non uniforme 
fltrm=[1 2 1;2 4 2;1 2 1];  % creation de filtre moyonneur par la méthode analytique 
fltrm=fltrm * (1/16);        %division sur 16 pour avoir la moyenne. 
I1=imfilter(I,fltrm);   % appliquer le filtre crée sur l'image I.
% afichage des trois image pour les compareés.
figure(2), subplot(1,3,1), imshow(I); title('l"image original');
subplot(1,3,2), imshow(Iav);title('l"image filtrer par filtre moyenneur uniforme');
  subplot(1,3,3), imshow(I1);title('l"image filtrer par le filtre moynneur non uniforme '); 
%%
%creation de notre propre filtre 
f=[2 6 2;6 4 6;5 8 5];   %notre cofficients qu'on a choisi.
f=f * (1/44);            % division sur 44 la somme des coefficients
I2=imfilter(I,f);        % appliquer le filtre sur l'image I
%affichage des deux images
figure, subplot(1,2,1), imshow(I); title('l"image original');
  subplot(1,2,2), imshow(I2);title('l"image filtrer I2');
  
 %On remarque que plus la taille de la zone considérée pour la moyenne est grande, plus l'effet de lissage est important. Donc le résultat ça déponds des coefficients qu’on a donner pour le filtre. 
%%
fgauss=fspecial('gaussian',9,2.5); %filtre gaussian 
figure, bar3(fgauss,'b'); title('le filtre gaussian en 3D')   % trace de filtre en 3D.
Igauss=imfilter(I,fgauss); %appliquer le filtre sur l'image I
% affichage des résultats
figure, subplot(1,2,1), imshow(I); title('l"image original');
subplot(1,2,2), imshow(Igauss);title('l"image filtrer par le filtre gaussian ');
%%
flap=fspecial('laplacian',0 ); % le filtre laplacian avec une somme =0
II=double(I); % convertir de uint8 vers double
IIfltrii=imfilter(I,flap); %appliquer le filtre.
%affige des resultats.
figure, subplot(1,2,1), imshow(I); title('l"image original');
subplot(1,2,2), imshow(IIfltrii);title('l"image filtrer par le filtre laplacian ');

%% FILTRAGE FREQUENTIEL 
IMG=imread('maison.jpg');
c=5;
Y=fft2(IMG); % transformer de fourier 
Y=fftshift(Y);  % pour organise les fréquence nulles au centre et les HF en extrimeté 
%Deux façons de calculer le module :
Y1=abs(Y);               %Echelle linéaire.
Y2= 10*log10(c+abs(Y));  % Echelle logarithmique.

%pour normalisé Y renormalise la matrice en fixant le minimum  à 0 et le
%maximum à 1.
Y1_norm=mat2gray(Y1);
Y2_norm=mat2gray(Y2);
%afficher le resultats.
figure, subplot(2,3,1), imshow(IMG), title("image originale");
subplot(2,3,2), imshow(Y1_norm), title('echelle linéaire ');
subplot(2,3,3), imshow(Y2_norm) , title('echelle logarithmique ');

% transformer de fourier inverse 
T_inv_F=real(ifft2(Y));
subplot(2,3,4), imshow(T_inv_F,[]) , title('image TF inverse  ');

T_inv_F_nor=mat2gray(T_inv_F);   % normalisé le TFI de l'image.

 subplot(2,3,5:6),imshow(T_inv_F_nor,[]), title("image résultante apres TFI");  % retrouver l'image originale.

%%
part 2:
%%
X1=imread('cameraman.tif');


% b- l'image contient plus des bases fréquences , oui il y'a des des directions privilégiées 
% C)AND D)
Y1=fft2(X1);           %sans réarengement
Y1=fftshift(Y1);       % avec rearengement
XX1=255*real(ifft2(Y1));    % la partie réel de la TFI de l'image 
mod_ule=abs(Y1);                   %module dans l'echelle linéaire 
mod_ule1=10*log10(100+abs(Y1));    %module dans l'echelle logarithméque
Y1_norma=mat2gray(mod_ule);         % normalisation 
Y2_norma=mat2gray(mod_ule1);        % normalisation
figure(2),subplot(2,2,1),imagesc(X1), title('image originale');
subplot(2,2,2),imagesc(Y1_norma),  title('spectre en echelle linéaire');
subplot(2,2,3),imagesc(Y2_norma),  title('spectre en echelle logarithemique');
subplot(2,2,4),imshow(XX1,[]), title('image reconstruit après la TFI');
% E) l'échelle logarithmique est le mieux.
% F) comme un commantaire on peut dire que l'image contient plus des BF qui
% localisé au centre de l'image ce que on peut voire facillement en spectre
% de l'echelle logaritmique en plus il y'a des directions en extremité.


%part 3:
%% Création des 4 images
for i=1:100
    for j=1:100
fx=0.1;fy=0.4; 
f1(i,j)=sin(fx*i+fy*j);
f2(i,j)=sin(fx*j);
f3(i,j)=0.5*(sin(fx*i)+sin(fy*j));
f4(i,j)=j/100;
    end 
end
C=100;
%% pour la f1
Y1=fft2(f1);  % transformer de fourier de f1
Y1=fftshift(Y1); % réarengement de Y1
Y_lin=abs(Y1);             %module en  l'echelle linéaire.
Y_log=10*log(100+abs(Y1)); %module en  l'echele logarithmique.

Y_lin_norm=mat2gray(Y_lin);    %normalisation 
Y_log_norm=mat2gray(Y_log);     %normalisation 
% affichage des résultats.
figure(1), subplot(131), imshow(f1);title('image originale');
subplot(132), imshow(Y_lin_norm);   title('spectre en echelle linéaire');
subplot(133), imshow(Y_log_norm);   title('spectre en echelle logarithemique');
%% pour la f2
Y2=fft2(f2);  % transformer de fourier de f2
Y2=fftshift(Y2); % réarengement de Y1
Y_lin2=abs(Y2);             %l'echelle linéaire.
Y_log2=10*log(100+abs(Y2)); % l'echele logarithmique.

Y_lin_norm2=mat2gray(Y_lin2);    %normalisation
Y_log_norm2=mat2gray(Y_log2);    %normalisation
% affichage des résultats 
figure(2), subplot(131), imshow(f2);title('image originale');
subplot(132), imshow(Y_lin_norm2);   title('spectre en echelle linéaire');
subplot(133), imshow(Y_log_norm2);   title('spectre en echelle logarithemique');

%% pour la f3
Y3=fft2(f3);  % transformer de fourier de f3
Y3=fftshift(Y3); % réarengement de Y1
Y_lin3=abs(Y3);             %l'echelle linéaire.
Y_log3=10*log(100+abs(Y3)); % l'echele logarithmique.

Y_lin_norm3=mat2gray(Y_lin3);    
Y_log_norm3=mat2gray(Y_log3);
figure(3), subplot(131), imshow(f3);title('image originale');
subplot(132), imshow(Y_lin_norm3);   title('spectre en echelle linéaire');
subplot(133), imshow(Y_log_norm3);   title('spectre en echelle logarithemique');

%% pour la f4
Y4=fft2(f4);  % transformer de fourier de f4
Y4=fftshift(Y4); % réarengement de Y1
Y_lin4=abs(Y4);             %l'echelle linéaire.
Y_log4=10*log(100+abs(Y4)); % l'echele logarithmique. pour mieux voir le spectre.

Y_lin_norm4=mat2gray(Y_lin4);    
Y_log_norm4=mat2gray(Y_log4);
figure(4), subplot(131), imshow(f4);title('image originale');
subplot(132), imshow(Y_lin_norm4);   title('spectre en echelle linéaire');
subplot(133), imshow(Y_log_norm4);   title('spectre en echelle logarithemique');

%%
part 4%%

%% B-1
Z1=imread("C:\Users\HANDAG\OneDrive\Images\oldcar2.png");
Z1=rgb2gray(Z1);
figure, subplot(3,3,1), imshow(Z1,[]) ,title('img original ');
masque=SeuillageFrequencesFourier(Z1,70,1);
subplot(3,3,2), imshow(masque,[]) ,title('masque');
% on remarque que on a crée une image noire avec une cercle blanche au
% centre son rayon est R=20
% B-2 
X1=ifftshift(fft2(Z1)); %avoir le TF et les BF soient en centre.
X1_fil=X1.*masque;%Applique le masque aux TF de l'image X1 pour filtrer les hautes fréquences.
X1_fil=ifft2(ifftshift(fft2(X1_fil))); %faire le TFI pour revenir à l'espace de pixeles.
mod_lin=abs(X1);     % module en l'echelle linéaire
mod_lin_fil=abs(X1_fil); % module en l'echelle linéaire.
mod_logar=10*log(10+abs(X1)); % module en l'echelle logarithmique
mod_logar_fil=10*log(10+mod_lin_fil); % module en l'echelle logarithmique.


subplot(3,3,3), imshow(mat2gray(mod_lin),[]) ,title("l'ech linéaire de la TF DE img original ");
subplot(3,3,4), imshow(mat2gray(mod_lin_fil),[]) ,title("l'ech linéaire d'img FIL");
subplot(3,3,5:6), imshow(mat2gray(mod_logar),[]) ,title("l'ech logarithmique de la TF DE img original");
subplot(3,3,7), imshow(mat2gray(mod_logar_fil),[]) ,title("l'ech logarithmique d'img fil");
%image reconstruite 





%%
masque=SeuillageFrequencesFourier(Z1,20,1);
figure, imshow(masque) %on a crée une image blanche avec une cercle noire au
% centre son rayon est R=20

% B-3  si on change R le rayon de cerle et varié 
%% C /
Z1=imread("C:\Users\HANDAG\OneDrive\Images\img9.jpg");

masque=SeuillageFrequencesFourier(Z1,20,1);
figure, imshow(masque), title('image resultante')

%meme resultats meme si on change l'image on trouve une image avec une
%cercle au centre la seule qui change est la taille d'image 



% part 5:

%% 8. Filtrage d’une image : domaine fréquentiel
X=imread('cameraman.tif'); % Lire l'image 
X=double(X)/255; %Convertit les valeurs de l'image en double  et les normalise entre 0 et 1.

h=fspecial('averag',[15 15]);   % creation de filtre moyenneur 
X_fltrer=imfilter(X,h);      % application de filtre 
[n m]=size(X);

% Calculer la transformation de Fourier 2D de l'image et le filtre
fft_X=fft2(X,n,m); % Calcule la FFT du img X avec les mêmes dimensions que l'image X.
fft_h=fft2(h,n,m); %  Calcule la FFT du filtre h avec les mêmes dimensions que l'image X.

Y_fil=fft_X.*fft_h;  %Multiplie les spectres de Fourier de X et de h dans le domaine fréquentiel pour appliquer le filtrage.
Ffrq=mat2gray(real(ifft2(Y_fil)));% Le TFI de spectre Y_fil puis normalisé.
figure(1), subplot(3,2,1), imshow(X),title("img original");
subplot(3,2,2), imshow(X_fltrer);title("L'image filtré");
subplot(3,2,3), imshow(abs(ifftshift(fft_h)));title("l'éch linéaire du spectre de Fourier de h.");
subplot(3,2,4), imshow(Y_fil);title("img filtré en freq");
subplot(3,2,5), imshow(Ffrq);title("l'image restaurée.");


%%
EQM1=1/n/m*sum(sum((X-X_fltrer).^2))%Calcule l'erreur quadratique moyenne entre l'image originale X et l'image filtrée X_fltrer.
PSNRsp=10*log10(1/EQM1)%Calcule le rapport signal sur bruit (PSNR) spatial.
EQM2=1/n/m*sum(sum(X-Ffrq).^2)%Calcule l'erreur quadratique moyenne entre l'image originale X et l'image restaurée Ffrq.
PSNRfrq=10*log10(1/EQM2)%Calcule le rapport signal sur bruit (PSNR) fréquentiel.
%%
%fft_X1=fftshift(fft_X1);
% calacule de fft de image filtré 
%X1_fft=fftshift(fft2(X1_fltrer));
%mod_X1_fft=10*log(1+abs(X1_fft));
%X1_ftt_nor=mat2gray(mod_X1_fft);

% Afficher le spectre de l'image originelle
%mod_X1=10*log(1+abs(fft_X1));
%mod_nor=mat2gray(mod_X1);
%figure(5), imshow(mod_nor,[]);

% Afficher le spectre de l'image filtrée
%figure(6);
%imshow(X1_ftt_nor);
%title('Spectre de l''image filtrée');

%la réponse en fréquence du filtre.
%rep_freq=fftshift(fft2(h));
%figure(7), imshow(rep_freq,[]), title('la réponse en fréquence du filtre.');

%% pour un filtre gaussian 
X2=imread('cameraman.tif'); % Lire l'image 
X2_nor=mat2gray(X2);         % renormalisé l'image
h2=fspecial('gaussian',[15 15],1.5);   % creation de filtre gaussian
X2_fltrer=imfilter(X2,h);      % application de filtre 


% Calculer la transformation de Fourier 2D de l'image
fft_X2=fft2(X2);
fft_X2=fftshift(fft_X2);

% calacule de fft de image filtré 
X2_fft=fftshift(fft2(X2_fltrer));
mod_X2_fft=10*log(1+abs(X2_fft));
X2_ftt_nor=mat2gray(mod_X2_fft);

% Afficher le spectre de l'image originelle
mod_X2=10*log(1+abs(fft_X2));
mod_nor=mat2gray(mod_X2);
figure(5), imshow(mod_nor,[]);

% Afficher le spectre de l'image filtrée
figure(6);
imshow(X2_ftt_nor);
title('Spectre de l''image filtrée');

%la réponse en fréquence du filtre.
rep_freq2=fftshift(fft2(h2));
figure(7), imshow(rep_freq2), title('la réponse en fréquence du filtre.');


%%
fgaus=fspecial('gaussian',9,1.5); %creation de filtre
figure , bar3(fgaus,'b')          % répresenté le filtre sous forme des bar 3D.
L=imfilter(X,fgaus); %appliquer le filtre.

figure,
subplot(131),imshow(X), title("image originale");
subplot(132),imshow(L), title("image filtré");

fgaus1=fspecial('gaussian',9,0.3);
LL=imfilter(X,fgaus1);     %appliquer le filtre.
subplot(133),imshow(LL), title("image filtré 2");

%%


















