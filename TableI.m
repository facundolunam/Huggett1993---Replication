%------------------------------------------------------------
% Author:  Facundo Luna - Rutgers University
% email: facundo.luna@rutgers.edu
% This code replicates Table I in Huggett (1993)
% To replicate Table II, just change sigma=3.
%------------------------------------------------------------


%---------------------House Keeping--------------------------
clear; close all; clc;
tic
%% Parameter
beta    = 0.99322;              % Discount Factor                        
sigma   = 1.5;                  % Risk Aversion Parameter                                                               
CreditLimit=[-2; -4; -6; -8];   % Credit Limits
amax    = 15;                   % Max asset level
na=1000;                         % Grid Points for Assets                                
e       = [1 0.1];              % States                                                   
PI   = [.925 .075; .5 .5];      % Transition Matrix                                           
ns      = size(PI,2);           % Number of states

% Preferences
u = @(c) c.^(1-sigma)/(1-sigma); 

% Matrix to store results
InterestRate=zeros(4,1);
Price=zeros(4,1);
tol     = 10^-4; % Tolerance of error

parfor jj=1:4                                                 
amin= CreditLimit(jj);      % Borrowing constraint
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
InterestRate(jj)=100*((1/q)^6-1);
Price(jj)=q;
end


%% Table 1 in Hugget (1993)
Table = table(CreditLimit,InterestRate,Price)
toc





