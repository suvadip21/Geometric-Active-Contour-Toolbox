function [inter_res] = runChanVese(obj,nu,display_interval)
%RUNCHANVESE Segmentation using Chan-Vese's method

phi = obj.phi0;
n_its = obj.max_its;
I     = obj.Img;
magnify = obj.mag;
thold = obj.thresh;
col = obj.color;

figure(2);
[evolved_phi,its,inter_res] = segmentChanVese(I,n_its,phi,nu,thold,magnify,col,display_interval);
obj.phi = evolved_phi;
obj.convg_steps = its;

end


function [phi,its,inter_res] = segmentChanVese(u,max_iter,phi, reg_term,convg_err,mag,color,display)

Dirac_global            = @(x,e) ((e/pi)./(e^2.+ x.^2));
Heaviside               = @(y,e) (0.5*(1+(2/pi)*atan(y/e)));

count_lim   = 20;

stop = 0;
count = 0;
frame = [];
its  = 0;
inter_res = [];
while (its < max_iter && stop == 0)
    
    h_phi = Heaviside(phi,2);
    inside_mask = h_phi;
    outside_mask = 1-h_phi;
    u_in = u.*inside_mask;
    u_out = u.*outside_mask;

    c1 = sum(u_in(:))/(sum(inside_mask(:)));
    c2 = sum(u_out(:))/(sum(outside_mask(:)));
       
    div_term   = GeomAC.compute_divergence(phi);
    delta_phi   = Dirac_global(phi,2);
    
    evolve_force = delta_phi.*(-(u-c1).^2 + (u-c2).^2);
    reg_force    = reg_term*div_term;
    
    dphi_dt = evolve_force./(max(abs(evolve_force(:)))+eps) + reg_force;
    delta_t = .6/(max(abs(dphi_dt(:)))+eps);          % Step size using CFL
    
    
    prev_mask = phi >=0;
    
    phi = phi + delta_t*dphi_dt;
    phi = GeomAC.SussmanReinitLS(phi,0.5);
    phi = GeomAC.NeumannBoundCond(phi);
    if display > 0
        
        if mod(its,display) == 0
            inter_res=cat(3,inter_res, phi);
             GeomAC.showCurveAndPhi(phi,mag,u,color);
            drawnow;
        end
    end
    
    
    curr_mask = phi >=0 ;
    count = GeomAC.convergence(prev_mask,curr_mask,convg_err,count);
    % count how many succesive times we have attained convergence, reduce local minima
    if count <= count_lim
        its = its + 1;
    else
        stop = 1;
        fprintf('Algorithm converged, iteration=%d \n',its);
    end
  
    
end

if stop == 0
    fprintf('Did not converge in %d steps\n',its);
end

inter_res=cat(3,inter_res, phi);        
GeomAC.showCurveAndPhi(phi,mag,u,color);
end



