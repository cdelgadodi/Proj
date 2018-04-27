function [Xk, gk, i] = conjugate_gradient(f, Xk, options)

    if (~exist('options','var'))
        options.MaxIter = 50;
        options.epsilon = 0.01;
        options.tau = 1;
    end    
    
    i = 0;    
    [fk,gk] = f(Xk); %Theta c
    dk = - gk;
    i = 1; % Es el k
    Xk = Xk + options.tau*dk; %theta_k+1 = Theta_k + tau*dk
    gkp = gk; %guardar el gradiente anterior para calcular el beta
    [fk,gk] = f(Xk);
    beta = (gk*(gk-gkp)')/(norm(gkp))^2;
    k = 0;
    %-- Main loop
    while ( (norm(gk)/norm(fk) > options.epsilon) && (i<options.MaxIter) )

        %-- internal iterator
        i = i+1;
        [fk,gkp] = f(Xk);
        if k == 1
            dk = - gkp;
            Xk = Xk + options.tau*dk;
        else
            dk = -gkp + beta*dk;
            Xk = Xk + options.tau*dk;
        end 
        %-- display iterative cost value
        S = 'Iteration ';
        f1 = norm(fk);
        fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
        
    end
    
    fprintf('\n');
    
    
    
    
end