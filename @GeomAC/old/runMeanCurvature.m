function [] = runMeanCurvature(obj,sigma,c,display)
%% Malladi-Sethian:
% $$\phi_t = sgn*(c+\kappa)g_{\sigma}(I)|\nabla \phi| $$ 
% $sgn > 0$ : curve moves outwards, $sgn < 0$: curve moves inwards


phi = obj.phi0;
max_iter = obj.max_its;
I     = obj.Img;
magnify = obj.mag;
thold = obj.thresh;
col = obj.color;

its = 0;
stop = 0;
count = 0;
p = 1;
count_lim  = 20;


filt = fspecial('gaussian',6*sigma,sigma);
u = conv2(I,filt,'same');
[ux,uy] = gradient(u);

g = 1./(0.001+(sqrt(ux.^2+uy.^2)).^p);


while (its < max_iter && stop == 0)
    
    kappa = lsfCurvature(phi);
    [phix,phiy] = gradient(phi);
    grad_phi = sqrt(phix.^2+phiy.^2);
    
    dphi_dt1 = g.*grad_phi;          % Area based propagation
    dphi_dt2 = g.*grad_phi.*kappa;   % curvature based term
    
    dphi_dt1 = dphi_dt1/(eps+max(abs(dphi_dt1(:))));
    dphi_dt2 = dphi_dt2/(eps+max(abs(dphi_dt2(:))));
    
    
    F = c*dphi_dt1 + dphi_dt2;
    dt = 0.9/max(eps+abs(F(:)));
    
    prev_mask = phi >=0 ;
    phi = phi+dt*F;
    phi = GeomAC.SussmanReinitLS(phi,0.5);
    phi = GeomAC.NeumannBoundCond(phi);
    
    if display > 0
        if mod(its,display) == 0
            GeomAC.showCurveAndPhi(phi,magnify,I,col);
            drawnow;
        end
    end
    
    curr_mask = phi >=0 ;
    count = GeomAC.convergence(prev_mask,curr_mask,thold,count);
    
    if count <= count_lim
        its = its + 1;
    else
        stop = 1;
        fprintf('Algorithm converged, iteration=%d \n',its);
    end
    
end

if stop == 0
    fprintf('End of iteration %d\n',its);
end


GeomAC.showCurveAndPhi(phi,magnify,I,col);

end



function k = lsfCurvature(u)

f = fspecial('gaussian',6,2);
u = imfilter(u,f,'same');
[ux,uy] = gradient(u);
mag = eps+sqrt(ux.^2+uy.^2);
uxx = gradient(ux./mag);
[~,uyy] = gradient(uy./mag);
k = uxx+uyy;

end




