%------------------------------------------------------------
% Author:  Facundo Luna - Rutgers University
% email: facundo.luna@rutgers.edu
% This code replicates Figure I and II in Huggett (1993)
%------------------------------------------------------------

%---------------------House Keeping--------------------------
clear; close all; clc;
tic
%% Parameter
beta    = 0.99322;              % Discount Factor                        
sigma   = 1.5;                % Risk Aversion Parameter                                                               
amin=-2;                        % Credit Limits
amax    = 15;                   % Max asset level
na=1000;                        % Grid Points for Assets                                
e       = [1 0.1];              % States                                                   
PI   = [.925 .075; .5 .5];      % Transition Matrix                                           
ns      = size(PI,2);           % Number of states

% Preferences
u = @(c) c.^(1-sigma)/(1-sigma); 

tol     = 10^-4; % Tolerance of error

a= linspace(amin,amax,na);  % Assets Grid                                          

% Initial Guess for q
q       = 1;                                                                 
upper   = 1.1;                                                                   
lower   = beta; 
ED=1; iter=1;

% Pre-allocate matrices  
[anext_grid, s_grid, a_grid] = ndgrid(1:na, 1:ns, 1:na);
vnext = zeros(ns, na);
apolicy = zeros(ns, na); 
v=zeros(ns,na); 

%% Iteration to compute Price (q)
while abs(ED) >tol && iter<50

% Calculate Consumption Matrix and Utility
c = a(a_grid) + e(s_grid) - q * a(anext_grid);
c(c<=0)=10^-6;                                                                                       
Umat =u(c);

%% Value Function Iteration
diff=1;
while diff > tol
    % Value Function Matrix
temp1 = Umat(:,1,:) + beta * PI(1,1) * v(1, :)' + beta * PI(1,2) * v(2, :)';
temp2 = Umat(:,2,:) + beta * PI(2,1) * v(1, :)' + beta * PI(2,2) * v(2, :)';
Vmat=[temp1, temp2];

    for aa = 1:na
          [vnext(1,aa), apolicy(1,aa)] = max(Vmat(:, 1, aa),[],1);
          [vnext(2,aa), apolicy(2,aa)] = max(Vmat(:, 2, aa),[],1);
    end

    diff = max(max(abs(vnext-v)));     
    % Use McQueen-Porteus Error Bounds algorith to improve speed
    bl=(beta/(1-beta))*(min(min(vnext-v)));            
    bh=(beta/(1-beta))*(max(max(vnext-v)));     
    v=vnext+(bh+bl)/2;      
end
%% Stationary Distribution
   % Initial Guess
   Psi=zeros(ns,na); Psi(1)=1;
   Diff=1;
  
   while Diff > tol
    Psinext=zeros(ns,na);
        for j=1:na
            Psinext(1,apolicy(1,j)) =  Psinext(1,apolicy(1,j)) + PI(1,1)*Psi(1,j);
            Psinext(2,apolicy(1,j)) =  Psinext(2,apolicy(1,j)) + PI(1,2)*Psi(1,j);
            Psinext(1,apolicy(2,j)) =  Psinext(1,apolicy(2,j)) + PI(2,1)*Psi(2,j);
            Psinext(2,apolicy(2,j)) =  Psinext(2,apolicy(2,j)) + PI(2,2)*Psi(2,j);
        end
    
    Diff=abs(max(max(Psinext-Psi)));
    Psi=Psinext;
   end
   
%% Compute Excess of Demand   
ED = sum(sum(a(apolicy).*Psi));

%% Update Price (q)
    if ED<0          
        upper=q;
    else
        lower=q;
    end
q=(lower+upper)/2;  
iter=iter+1;

end


%% Figure I
plot(a,a(apolicy)) %Policy Functions
hold on
plot(a,a) % 45 degree Line
legend('eh', 'el', '45 degree')
title('Figure 1: Optimal Decision Rule')
hold off

%% Figure II
plot(a,cumsum(Psi(1,:)), a,cumsum(Psi(2,:))) %Ploicy Functions
legend('eh', 'el')
title('Figure 2: Stationary Distribution')





