clc
clear all
close all

% x=imread('data/train/0img/001.png');
% figure(1)
% imshow(x)


% [data, path] = uigetfile('*.*');
% x = imread(strcat(path, data));

 x=imread('data/train/0img/001.png');
%img8 = uint8(x/256); x=img8;
%img8= im2uint8(x); x = img8;

mu = round(mean(x(:))*2.55 )
xx=x;   xx(x>mu)=mu; 
x_ = (mu-xx);


figure(1)
imshow(x_,[])


K = imadjust(x,[0.0 .5],[]);
figure
imshow(K)




%img8 = uint8(x/256); x=img8;
img8= im2uint8(x); x = img8;

mu = round(mean(x(:))*0.35 )
%mu =45
xx=x;   xx(x>mu)=mu; 
x_ = (mu-xx);
%x_=x_/max(x_(:));
% figure(2)
% imshow(x_, [])















% figure(2)
% imshow(x)

% J = histeq(xx);
% figure,imshow(J)
% figure,imhist(x)