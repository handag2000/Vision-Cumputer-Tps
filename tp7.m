clear all; clc; close all
%% seuillage manuel
a = imread('coins.png'); % lire l'images 
% tracer l'image et so histogramme.
figure(1), subplot(121), imshow(a),title('image originale'); 
subplot(122), imhist(a),title('histo de a '), axis([0 255 0 4500]); % affiché histogramme de l'img originale

b=im2bw(a,0.35); %changer le seuil 

figure(2), subplot(121), imshow(b),title('image àpres l"application de seuillage');
subplot(122), imhist(b),title('histo de b '), axis([0 255 0 60000]);

%% seuillage automatique  algoritheme de otsi

level = graythresh(a); % claculer la seuille.
b1=im2bw(a,level);

figure(3), subplot(121), imshow(b1),title('image avec seuillage automatique');
subplot(122), imhist(a),title('histo de b1 '), axis([0 255 0 60000]);

%% utilisation de la fonction bwtraceboundary( ) pour tracer la bordure d'un objet.

%définir le poit de départ pour le suivi le conteur.
dim = size(b1); % la taille de b1.
col= round(dim(2)/2)-90; %on calcule la colonne de départ pour le suivi du contour.
row = min(find(b1(:,col)));% on cherche la position de premier ligne non nulle de col dans l'image b1.

boundary=bwtraceboundary(b1,[row,col],'N');% suivre le conteur dans sens Nord avec une connectivité de 2
%affiche le conteur sur l'image 
figure(4);imshow(a); hold on; % utilation de hold on pour ajouté le trace de conteur sur l'image.
plot(boundary(:,2),boundary(:,1),'g','LineWidth',3);

% NB : on utilise impixel pour avoir la position de zone qu'on veut
% contourer  EX impixel(a).



%% split and mearge.
a = imread('coins.png');

S = qtdecomp(a,.17);%effectue une décomposition en quadrants de l'image a.
blocks = repmat(uint8(0),size(S)); % Créer des blocs vides.
for dim = [256 128 64 32 16 8 4 2 1]
numblocks= length(find(S==dim));
if (numblocks > 0)
values = repmat(uint8(1),[dim dim numblocks]);%crée une table remplie par les 1 est de taille dim*dim*numblocks.
values(2:dim,2:dim,:)=0; %on donne pour tt les pixels intérieur de chaque blocs la val 0. en laissant les extrimité à 1.
blocks = qtsetblk(blocks,S,dim,values);% insérer les blocs spécifiés par values dans la matrice blocks.
end
end
blocks(end,1:end) =1; blocks(1:end,end) = 1;% définissent les pixels de bordure.
subplot(1,2,1), imshow(a);
k=find(blocks==1); %Rechercher les pixels de bordure des régions.
A=a; A(k)=255; %Superposer à l’image originale.
subplot(1,2,2), imshow(A);
%%











