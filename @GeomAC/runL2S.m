function [inter_res] = runL2S(obj,n_poly, nu,l2_reg, display_interval)
%RUNCHANVESE Segmentation using Chan-Vese's method

phi = obj.phi0;
n_its = obj.max_its;
I     = obj.Img;
magnify = obj.mag;
thold = obj.thresh;
col = obj.color;

basis_vect = LegendreBasis2D_vectorized(I,n_poly);

figure(2);
[evolved_phi,its,inter_res] = segmentL2S(I,basis_vect,n_its,phi,nu,thold,l2_reg, magnify,col,display_interval);
obj.phi = evolved_phi;
obj.convg_steps = its;

end


function [phi,its,inter_res] = segmentL2S(u,B,max_iter,phi, reg_term,convg_err,lambda_l2, mag, color,display)

Dirac_global            = @(x,e) ((e/pi)./(e^2.+ x.^2));
Heaviside               = @(y,e) (0.5*(1+(2/pi)*atan(y/e)));

count_lim   = 20;

stop = 0;
count = 0;
frame = [];
its  = 0;
inter_res = [];

II = eye(size(B,2),size(B,2));

while (its < max_iter && stop == 0)
    
    h_phi = Heaviside(phi,2);
    inside_mask = h_phi;
    outside_mask = 1-h_phi;
    u_in = u.*inside_mask;
    u_out = u.*outside_mask;
    
    u_in = u_in(:);
    u_out = u_out(:);
    
    inside_indicator = inside_mask(:);
    outside_indicator = outside_mask(:);
    
    A1 = B';                                            % (each row contains a basis vector)
    A2 = A1.*(repmat(inside_indicator',size(A1,1),1));   % A1, with each row multiplied with hphi
    B2 = A1.*(repmat(outside_indicator',size(A1,1),1));   % A1, with each row multiplied with hphi
    
    c1_vec = (A1*A2' + lambda_l2*II)\(A1*u_in);
    c2_vec = (A1*B2' + lambda_l2*II)\(A1*u_out);
    
    p1_vec = B*c1_vec;
    p2_vec = B*c2_vec;

    p1 = reshape(p1_vec,size(u));
    p2 = reshape(p2_vec,size(u));

    div_term    = GeomAC.compute_divergence(phi);
    delta_phi   = Dirac_global(phi,2);
    
    evolve_force = delta_phi.*(-(u-p1).^2 + (u-p2).^2);
    
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
    fprintf('End of iteration %d\n',its);
end

inter_res=cat(3,inter_res, phi);       
GeomAC.showCurveAndPhi(phi,mag,u,color);
end



% Generate the orthonormal Legendre bases (vectorized, see Kale and Vaswani)
function [B] = LegendreBasis2D_vectorized(Img,k)

%LEGENDREBASIS compute K shifted legendre basis for the vectorized image

[Nr,Nc] = size(Img);
N = length(Img(:));     % Vectorized image

B = zeros(N,(k+1).^2);
[B_r,B_r_ortho] = legendre_1D(Nr,k);
[B_c,B_c_ortho] = legendre_1D(Nc,k);

ind = 0;
for ii = 1 : k+1
    for jj = 1 : k+1
        ind = ind+1;
        row_basis = B_r(:,ii);
        col_basis = B_c(:,jj);
        outer_prod = row_basis*col_basis';  % same size as the image
        B(:,ind) = outer_prod(:);
        
    end
end

end



function [B,orthonormal_B] = legendre_1D(N,k)

X = -1:2/(N-1):1;
p0 = ones(1,N);


B = zeros(N,k+1);
orthonormal_B = B;
B(:,1) = p0';
orthonormal_B(:,1) = B(:,1)/norm(B(:,1));

for ii = 2 : k+1
    Pn = 0;
    n = ii-1;   % degree
    for k = 0 : n
       Pn = Pn +  (nchoosek(n,k)^2)*(((X-1).^(n-k)).*(X+1).^k);
    end
    B(:,ii) = Pn'/(2)^n;
    orthonormal_B(:,ii) = B(:,ii)/norm(B(:,ii));
end

end





