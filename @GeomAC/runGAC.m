function [inter_res] = runGAC(obj,sigma,c,mu,display)
%% Geodesic Active Contour:
% $$ \phi_t = (c + \kappa)g_{\sigma}(I)|\nabla \phi| + \mu<\nabla g,\nabla \phi>. $$
% $$ c > 0 $$ : curve moves outwards, $$ c < 0 $$: curve moves inwards.
% $$ \mu=0 $$ is the mean curvature motion. 
% This algorithm uses the narrowband procedure.
show_intermediate = 1;

phi         = obj.phi0;
max_iter    = obj.max_its;
I           = obj.Img;
magnify     = obj.mag;
thold       = obj.thresh;
col         = obj.color;

its         = 0;
stop        = 0;
count       = 0;
p           = 1;
count_lim   = 10;


filt = fspecial('gaussian',6*sigma,sigma);
u = conv2(I,filt,'same');
[ux,uy] = gradient(u);

g = 1./(0.01+(sqrt(ux.^2+uy.^2)).^p);
[gx,gy] = gradient(g);
inter_res = [];
% g=10;
while (its < max_iter && stop == 0)
    
    
    idx = find(phi <= 1.2 & phi >= -1.2);   %-- get the curve's narrow band
    if ~isempty(idx)
 
        kappa = lsfCurvature(phi);
        [phix,phiy] = gradient(phi);
        grad_phi = sqrt(phix.^2+phiy.^2);
        
        dphi_dt1 = g.*grad_phi;           % Advection term
        dphi_dt2 = g.*grad_phi.*kappa;    % curvature based term
        dphi_dt3 = (gx.*phix+gy.*phiy);   % edge attraction term
        
        dphi_dt1 = dphi_dt1/(eps+max(abs(dphi_dt1(:))));
        dphi_dt2 = dphi_dt2/(eps+max(abs(dphi_dt2(:))));
        dphi_dt3 = dphi_dt3/(eps+max(abs(dphi_dt3(:))));

        F = c*dphi_dt1 + dphi_dt2 + mu*dphi_dt3;
        dt = 0.8/(eps+max(abs(F(:))));
        
        prev_mask = phi >=0 ;

        phi(idx) = phi(idx)+dt*F(idx);
        
        phi = GeomAC.SussmanReinitLS(phi,0.5);
        phi = GeomAC.NeumannBoundCond(phi);
        curr_mask = phi >=0 ;
        if display > 0
            if mod(its,display) == 0
                inter_res=cat(3,inter_res, phi);
                if its == 1
                    GeomAC.showCurveAndPhi(phi,magnify,I,'--r');
                else
                    GeomAC.showCurveAndPhi(phi,magnify,I,col);
                end
                drawnow;
            end
        end
        
        count = GeomAC.convergence(prev_mask,curr_mask,thold,count);
        
        if count <= count_lim
            its = its + 1;
        else
            stop = 1;
            fprintf('Algorithm converged, iteration=%d \n',its);
        end
    else
        break;
        
    end
end
inter_res=cat(3,inter_res, phi);
GeomAC.showCurveAndPhi(phi,magnify,I,col);
obj.phi = phi;
obj.convg_steps = its;

end



function k = lsfCurvature(u)

f = fspecial('gaussian',6,2);
u = imfilter(u,f,'same');
[ux,uy] = gradient(u);
mag = eps+sqrt(ux.^2+uy.^2);
[uxx,uxy] = gradient(ux);
[uyx,uyy] = gradient(uy);

k = (uy.^2.*uxx+ux.^2.*uyy - ux.*uy.*uxy - ux.*uy.*uyx)./(mag).^3;
end




