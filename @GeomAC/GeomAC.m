classdef GeomAC < handle
    %GEOMAC Class of Geometric Active contours
    
    properties
        Img             % Image to work on
        mag             % Image magnification factor (1 = 100)
        n_regions       % Number of regions to initialize
        mask_type       % 'rect', 'ellipse','multiball'
        init_reg        % initialized binary region
        max_its         % number of iterations before stopping evolution
        color           % color of the displayed contour, 'w','r','b','c'
        thresh          % threshold to determine automatic stopping
        phi0            % initialized sdf
        phi             % LSF
        convg_steps     % Iterations to converge
    end
    
       
    methods
        
    %-------------------------------------------------------------------------
    %---------------------  Non Static Methods   -----------------------------
    %-------------------------------------------------------------------------
    %                          Constructor
 
        function prop = GeomAC(Img,img_magnify,num_iter,contour_color,convg_thresh,  Type)
            prop.Img        = Img;
            prop.mask_type  = Type;
            prop.mag        = img_magnify;
            prop.max_its    = num_iter;
            prop.color      = contour_color;
            prop.thresh     = convg_thresh;
            
            prop.phi        = [];
            prop.init_reg   = [];
            prop.phi0       = [];
            prop.convg_steps = 0;
            
        end
        
        %-------------------------------------------------------------------------
        
        %          Function prototypes for remote file funcs.
        
        initRegions(obj)                                % Initialization of LSF             
        inter_res = runChanVese(obj,nu,disp_interval)               % Chan-Vese
        inter_res = runL2S(obj,pol_degree,nu,l2_reg,disp_interval)  % L2S
        inter_res = runGAC(obj,sigma,c,mu,display)        % Geodesic Active Contour
        inter_res = runEdgeRegion(obj,sigma,cv_coeff,length_reg,attr_coeff,display) % Edge assisted Chan Vese
        %-------------------------------------------------------------------------
        %      Create the signed distance function from the binary image
        
        function [] = computeSDF(obj)
            % inside >= 0, outside < 0
            bwI = obj.init_reg;
            lsf = bwdist(bwI)-bwdist(1-bwI)+im2double(bwI)-.5;
            obj.phi0 = -double(lsf); 
        end
        
    end
    
    %-------------------------------------------------------------------------
    %-----------------------  Static Methods   -------------------------------
    %-------------------------------------------------------------------------
    
    
    methods(Static)
        
        function c = convergence(p_mask,n_mask,thresh,c)
            diff = p_mask - n_mask;
            n_diff = sum(abs(diff(:)));
            if n_diff < thresh
                c = c + 1;
            else
                c = 0;
            end
        end
        
        %-------------------------------------------------------------------------
        %               Check boundary condition
        function g = NeumannBoundCond(f)
            
            [nrow,ncol] = size(f);
            g = f;
            g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
            g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
            g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
        end
        
        %-------------------------------------------------------------------------
        %          Reinitialize LSF by Sussman reinitialization method   
        
        function [D] = SussmanReinitLS(D,dt)
     
            %D  : level set function
            a = D - shiftR(D); % backward
            b = shiftL(D) - D; % forward
            c = D - shiftD(D); % backward
            d = shiftU(D) - D; % forward
            
            a_p = a;  a_n = a; % a+ and a-
            b_p = b;  b_n = b;
            c_p = c;  c_n = c;
            d_p = d;  d_n = d;
            
            a_p(a < 0) = 0;
            a_n(a > 0) = 0;
            b_p(b < 0) = 0;
            b_n(b > 0) = 0;
            c_p(c < 0) = 0;
            c_n(c > 0) = 0;
            d_p(d < 0) = 0;
            d_n(d > 0) = 0;
            
            dD = zeros(size(D));
            D_neg_ind = find(D < 0);
            D_pos_ind = find(D > 0);
            dD(D_pos_ind) = sqrt(max(a_p(D_pos_ind).^2, b_n(D_pos_ind).^2) ...
                + max(c_p(D_pos_ind).^2, d_n(D_pos_ind).^2)) - 1;
            dD(D_neg_ind) = sqrt(max(a_n(D_neg_ind).^2, b_p(D_neg_ind).^2) ...
                + max(c_n(D_neg_ind).^2, d_p(D_neg_ind).^2)) - 1;
            
            D = D - dt .* GeomAC.sussman_sign(D) .* dD;
        end
        
        
        function S = sussman_sign(D)
            S = D ./ sqrt(D.^2 + 1);
        end
        
        %-------------------------------------------------------------------------
        %               Compute curvature of a function u(lsf)
        
        function k = compute_divergence(u)
            % Computes div(\nabla u/|\nabla u|)
            [ux,uy] = gradient(u);
            normDu = sqrt(ux.^2+uy.^2+1e-10);	
            Nx = ux./normDu;
            Ny = uy./normDu;
            nxx = gradient(Nx);
            [~,nyy] = gradient(Ny);
            k = nxx+nyy;                        
        end
        
        
        %-------------------------------------------------------------------------
        %                   Display the evolving curve
        
        function showCurveAndPhi(phi,magnify,Img,cl)
            
            imshow(Img,[],'InitialMagnification',magnify);
            hold on;
            [c,h] = contour(phi,[0 0],cl,'Linewidth',3); hold off;
            test = isequal(size(c,2),0);
            while (test==false)
                s = c(2,1);
                if ( s == (size(c,2)-1) )
                    t = c;
                    hold on; plot(t(1,2:end)',t(2,2:end)',cl,'Linewidth',3);
                    test = true;
                else
                    t = c(:,2:s+1);
                    hold on; plot(t(1,1:end)',t(2,1:end)',cl,'Linewidth',3);
                    c = c(:,s+2:end);
                end
            end
        end
        
        
        function showIntermediate(phi,cl,sz)
           
            [c,h] = contour(phi,[0 0],cl,'Linewidth',sz);
            test = isequal(size(c,2),0);
            while (test==false)
                s = c(2,1);
                if ( s == (size(c,2)-1) )
                    t = c;
                    hold on; plot(t(1,2:end)',t(2,2:end)',cl,'Linewidth',sz);
                    test = true;
                else
                    t = c(:,2:s+1);
                    hold on; plot(t(1,1:end)',t(2,1:end)',cl,'Linewidth',sz);
                    c = c(:,s+2:end);
                end
            end
        end
        
    end
    
end
%-------------------------------------------------------------------------
%                       Utility functions

%-------------------------------------------------------------------------

function shift = shiftD(M)
shift = shiftR(M')';
end
%-------------------------------------------------------------------------

function shift = shiftL(M)
shift = [ M(:,2:size(M,2)) M(:,size(M,2)) ];
end
%-------------------------------------------------------------------------

function shift = shiftR(M)
shift = [ M(:,1) M(:,1:size(M,2)-1) ];
end
%-------------------------------------------------------------------------

function shift = shiftU(M)
shift = shiftL(M')';
end
%-------------------------------------------------------------------------






