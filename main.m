clc
clear all
clear classes
Img = imread('spine_noisy2.png');
if length(size(Img)) > 2
    Img = rgb2gray(Img);
end
% Img = 0.6*ones(100);
imshow(Img,'InitialMagnification',200);
%%
Img         = im2double(Img);
img_magnify = 200;
mask_type   = 'ellipse';
num_iter    = 500;
color       = 'g';
thresh      = 0;

length_reg  = 0.4;
display_interval = 50;
acObj = GeomAC(Img,img_magnify,num_iter,color,thresh,mask_type);
acObj.initRegions();                                % Initialize the level set
acObj.computeSDF();                                 % Compute the signed distance function
%%
% intermed_phi = acObj.runChanVese(length_reg,display_interval);     % regularization-factor,display-interval
intermed_phi = acObj.runL2S(1,length_reg,0,display_interval);      % # basis,regularization-factor, l2_regularizer, display-interval
% intermed_phi = acObj.runGAC(2,1,1,display_interval);          % sigma,advection_term (<0: inwards),attr_term,display
% acObj.runEdgeRegion(1,1,length_reg,1,display_interval);       % sigma,cv_coeff,length_reg,attr_coeff,display

%% Show Intermediate results

[~,~,l] = size(intermed_phi);
last = l-2;
figure(99);imshow(Img,'InitialMagnification',200); hold on;
for ii = 1 : 2: last
   if ii == 1       %-- Initial Contour
        GeomAC.showIntermediate(intermed_phi(:,:,ii),'g',4);
   elseif ii == last   %-- Final Contour
       GeomAC.showIntermediate(intermed_phi(:,:,ii),'y',2.5);
   else             %-- Intermediate output
       GeomAC.showIntermediate(intermed_phi(:,:,ii),'c',0.2);
   end
end
hold off;
%% Test Heaviside-Dirac-gradient relationship
Dirac_global            = @(x,e) ((e/pi)./(e^2.+ x.^2));
Heaviside               = @(y,e) (0.5*(1+(2/pi)*atan(y/e)));
phi = acObj.phi;
eps = 2;
h = Heaviside(phi,eps);
delta = Dirac_global(phi,eps);
[hx,hy]=gradient(h);
g_h = sqrt(hx.^2+hy.^2);

figure; 
subplot(1,3,1); imshow(h); title('H(\phi)');
subplot(1,3,2); imshow(delta); title('\delta(\phi)');
subplot(1,3,3); imshow(g_h); title('|\nabla H(\phi)|');




%%
delete(acObj);


