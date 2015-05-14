function [] = runEdgeRegion(obj,sigma,mu1,mu2,mu3,display)
% $$ C_t = \mu_1(-(u-c_1)^2+(u-c2)^2)\delta \phi + \mu_2div(\nabla \phi/|\nabla \phi|)+ \mu_3<\nabla \phi,\nabla g>  $$

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
count_lim   = 15;

Dirac_global = @(x,e) ((e/pi)./(e^2.+ x.^2));
Heaviside    = @(y,e) (0.5*(1+(2/pi)*atan(y/e)));


filt = fspecial('gaussian',6*sigma,sigma);
u = conv2(I,filt,'same');
[ux,uy] = gradient(u);

g = 1./(0.01+(sqrt(ux.^2+uy.^2)).^p);
[gx,gy] = gradient(g);



while (its < max_iter && stop == 0)
    
    
    idx = find(phi <= 1.2 & phi >= -1.2);   %-- get the curve's narrow band
    if ~isempty(idx)
        
        [phix,phiy]  = gradient(phi);
        h_phi        = Heaviside(phi,2);
        inside_mask  = h_phi;
        outside_mask = 1-h_phi;
        u_in         = I.*inside_mask;
        u_out        = I.*outside_mask;
        
        c1 = sum(u_in(:))/(sum(inside_mask(:)));
        c2 = sum(u_out(:))/(sum(outside_mask(:)));
        
        div_term   = GeomAC.compute_divergence(phi);
        delta_phi   = Dirac_global(phi,2);
        
        
        dphi_dt1 = delta_phi.*(-(I-c1).^2 + (I-c2).^2);
        dphi_dt2 = delta_phi.*div_term;
        dphi_dt3 = (gx.*phix+gy.*phiy);
        
        dphi_dt1 = dphi_dt1/(eps+max(abs(dphi_dt1(:))));
        dphi_dt2 = dphi_dt2/(eps+max(abs(dphi_dt2(:))));
        dphi_dt3 = dphi_dt3/(eps+max(abs(dphi_dt3(:))));
        
        F = mu1*dphi_dt1 + mu2*dphi_dt2 + mu3*dphi_dt3;
        
        dt = 0.6/max(eps+abs(F(:)));
        
        prev_mask = phi >=0 ;
        phi(idx) = phi(idx)+dt*F(idx);
        
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
    else
        break;
        
    end
end

GeomAC.showCurveAndPhi(phi,magnify,I,col);

obj.phi = phi;
obj.convg_steps = its;


end

