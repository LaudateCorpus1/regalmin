function [Z ActiveSet hist param] = regOmega(inputData,param,startingZ)
%tic;
% Active set algorithm to solve f(Z) + lambda Omega(Z)



%% param contains
%
% param.f : the function to minimize. 
%       (1) stands for L2 distance squared : f(Z) = .5 |Z-Y|^2 
%       (2) stands for multitask :           f(Z) = .5 |XZ - Y|_F^2 
%       (3) stands for quadratic regression  f(Z) = .5 | diag( X Z X' ) - y|_2^2
%       (4) stands for bilinear regression   f(Z) = .5 |X1 Z X2 - Y|_F^2 
%
% param.k param.q : block size parameters
% param.lambda : lagrange multiplier
%
%  ** OTHER PARAMETERS HAVE DEFAULT VALUES **
%
% param.PSD : true if the ouput is desired to be PSD, default false
% param.nbMainLoop : nb of runs of "Solve P_S", default = 5
% param.powerIter : nb runs power iteration, default = 500;
% param.stPtPowerIter : nb of different starting points to try
% param.epsStop : add a component to the working set when  varIJ >
% lambda*(1-param.epsStop)
%
% param.innerLoopIter : iterations of the prox-grad loop P_S default 50
% param.niterPS : number of times an active component is picked and the
% relative subproblem solved, default = 5


%% inputData contains
% inputData.X, inputData.Y X and Y used to compute the loss and the gradient

%% startingZ
% initial value of Z, for warm start, if not provided set to 0



%% OUTPUT 
% Z : the minimizer, a matrix of appropriate size
%
% ActiveSet : a list of elements containing
% ActiveSet : 
% ActiveSet.I : the sets suggested by the heuristic 
% ActiveSet.J : the sets suggested by the heuristic (right)
% ActiveSet.U : the pseudo-singular vectors (of size k)  
% ActiveSet.V : the pseudo-singular vectors (of size q)  
% ActiveSet.Sigma : the pseudo-singular values
% ActiveSet.Z : the matrices of k x q such that Z = sum_i Z{i}
% ActiveSet.tracenorm : the tracenorms of the Z{i}'s (may be equal to 0)
% ActiveSet.left ActiveSet.middle ActiveSet.right are build so that their product equals 
% the solution: 
% Z = ActiveSet.left*ActiveSet.middle*ActiveSet.right
%
%
% hist : a list of vectors containing the objective function values through
% iterations, the loss, penalty and duality gap



%%
if nargin < 3
    startingZ = setDefaultStartingZ(inputData,param);
end


Z = startingZ;
ActiveSet = {};
ActiveSet.I = {};
ActiveSet.J = {};
ActiveSet.U = {};
ActiveSet.V = {};
ActiveSet.Sigma = {};
ActiveSet.Z = {};
ActiveSet.tracenorm = {};
c = 1;
i = 0; 

param = setParamDefaultValues(param);

Objective = [];
Loss = [];
Pen = [];
DualityGap = [];

while c
    i = i+1;
    
    % solve problem P_S
  
    [Z ActiveSet obj loss pen dualityGap] = solvePS(Z,ActiveSet,param,inputData);
    H = gradient(Z,inputData,param); % gradient
    Objective = [Objective obj];
    Loss = [Loss loss];
    Pen = [Pen pen];
    DualityGap = [DualityGap dualityGap]; 
        [u v] = MultiTPI(-H,param);   % get a new descent direction using truncated power iteration   
        currI = find(u); currJ = find(v);
        varIJ = norm(H(currI,currJ));
    %    fprintf('variance = %2.4e, thresh = %2.4e\n',varIJ, param.lambda)
        if varIJ > param.lambda*(1-param.epsStop / param.k)   
           ActiveSet.I = [ActiveSet.I, currI];ActiveSet.J = [ActiveSet.J, currJ];
           ActiveSet.U = [ActiveSet.U, u(currI)]; ActiveSet.V = [ActiveSet.V, v(currJ)];ActiveSet.Sigma = [ActiveSet.Sigma, varIJ];
           ActiveSet.Z = [ActiveSet.Z, zeros(param.k,param.q)]; ActiveSet.tracenorm = [ ActiveSet.tracenorm , 0]; 
        else
            c = 0;
        end
    c = i<param.nbMainLoop & c;
end
ActiveSet = postProcessFactors(ActiveSet,Z);
hist.objective = Objective; 
hist.loss = Loss; 
hist.pen = Pen; 
hist.dualityGap = DualityGap; 
end



%%

%%%%%%%%%%%%%%%%%%%%%%%
% auxiliary functions %
%%%%%%%%%%%%%%%%%%%%%%%

%%

function [Z ActiveSet obj loss pen dualityGap] = solvePS(Z,ActiveSet,param,inputData)


cont = 1;
i = 0;
obj = [];
pen = [];
loss = [];
dualityGap = [];

   
%    loss0 = norm(inputData.Y-Z, 'fro')^2*.5;
%    pen0 = trace(Z);
%    obj0 = loss0 + param.lambda*pen0 ;
%   fprintf('loss = %2.6f, pen = %2.6f, obj = %2.6f \n',  loss0,pen0,obj0)
%  fprintf('zou\n')

while cont
    if isempty(ActiveSet.I)
        break;
    end
    i = i+1;
    ix = randi(length(ActiveSet.I),1); 
    currI = ActiveSet.I{ix};
    currJ = ActiveSet.J{ix};
    currPassiveZ = Z;
    currPassiveZ(currI,currJ) = Z(currI,currJ) - ActiveSet.Z{ix};
    loss = 0;
    if param.f == 1       % prox : the subproblem indexed by ix is solved in one iteration, a singular value shrinkage
        
        if param.PSD
            [currZ tracenorm u s] = ShrinkPSD(inputData.Y(currI,currI) - currPassiveZ(currI,currI) , param.lambda); v = u;          
        else
            [currZ tracenorm u v s] = Shrink(inputData.Y(currI,currJ) - currPassiveZ(currI,currJ) , param.lambda);
        end
    
    elseif param.f == 2   % multitask  
        currY = inputData.Y - inputData.X*currPassiveZ;        
        [currZ u v s grad tracenorm] = fista(inputData.X(:,currI),currY(:,currJ), param, ActiveSet.Z{ix});
      
    elseif param.f == 3   % quadratic       
        currY = inputData.Y - diag(inputData.X*currPassiveZ*inputData.X');
        [currZ u v s grad tracenorm] = fista(inputData.X(:,currI) , currY , param, ActiveSet.Z{ix});
    elseif param.f == 4   % bilinear       
        currY = inputData.Y - inputData.X1*currPassiveZ*inputData.X2;
        X{1} = inputData.X1(:,currI); 
        X{2} = inputData.X2(currJ,:);
        [currZ u v s grad tracenorm] = fista(X , currY , param, ActiveSet.Z{ix});

    end
    
    
    ActiveSet.Z{ix} = currZ;
    ActiveSet.U{ix} = u;
    ActiveSet.V{ix} = v;
    ActiveSet.Sigma{ix} = s;    
    Z = currPassiveZ;
    Z(currI,currJ) = Z(currI,currJ) + currZ;
    
    
    ActiveSet.tracenorm{ix} = tracenorm;
    if param.f == 1
        currloss =  .5*norm(Z - inputData.Y, 'fro')^2;
    elseif param.f == 2
        currloss =  .5*norm(inputData.X*Z - inputData.Y, 'fro')^2;
    elseif param.f == 3
        currloss = .5*norm(inputData.Y - diag(inputData.X*Z*inputData.X'))^2;
    elseif param.f == 4
        currloss = .5*norm(inputData.Y - inputData.X1*Z*inputData.X2, 'fro')^2;
    end
    
    loss(i) = currloss;
    pen(i) = sum(cell2mat(ActiveSet.tracenorm)) ;
    obj(i) = currloss + param.lambda*pen(i) ;
    dualityGap(i) = getDualityGap(Z,param,ActiveSet,inputData,obj(i));
  %  if mod(i,5)==1
   %     fprintf('loss = %2.6e, obj = %2.6e, duality = %2.6e, i = %d \n',  loss(i),obj(i),dualityGap(i), i)
  %  end
   cont = (dualityGap(i)>param.PSdualityEpsilon) && i< param.niterPS;

end

end




%%
function ActiveSet = postProcessFactors(ActiveSet,Z)


%% POST-PROCESSING ActiveSet
% this last step is a post-processing of the output in order to make it
% user-friendly / interpretable. The factors ActiveSet.left
% ActiveSet.middle ActiveSet.right are build so that their product equals 
% the solution: 
% Z = ActiveSet.left*ActiveSet.middle*ActiveSet.right ' 
outputRank = 0; 
for i = 1:length(ActiveSet.I)
     if ActiveSet.tracenorm{i}>0
        outputRank = outputRank + size(ActiveSet.U{i}, 2);
     end
end
ActiveSet.left = zeros(size(Z,1), outputRank);
ActiveSet.right = zeros(size(Z,2), outputRank);
dg = [];
ix = 0;
for i = 1:length(ActiveSet.I)
    if ActiveSet.tracenorm{i}>0
        ActiveSet.left(ActiveSet.I{i},(ix+1):(ix+size(ActiveSet.U{i},2))) = ActiveSet.U{i};
        ActiveSet.right(ActiveSet.J{i},(ix+1):(ix+size(ActiveSet.V{i},2))) = ActiveSet.V{i};
        dg = [dg; ActiveSet.Sigma{i}];
        ix = ix + size(ActiveSet.U{i},2);
    end
end

[dg o] = sort(dg,'descend');
ActiveSet.middle = diag(dg);
ActiveSet.left = ActiveSet.left(:,o);
ActiveSet.right = ActiveSet.right(:,o);
end

%%

function [uBest vBest lambdaBest] = MultiTPI(A,param)

lambdaBest = -inf;
uBest = randn(size(A,1),1); uBest = projectL0(uBest,param.q); uBest = uBest/norm(uBest);
vBest = randn(size(A,2),1); vBest = projectL0(vBest,param.q); vBest = vBest/norm(vBest);
       
for i=1:param.stPtPowerIter
    [u v] = truncatedPowerSeq(A,param);
    lambda = (u'*A*v);
    if lambdaBest < lambda
        uBest = u;
        vBest = v;
        lambdaBest = lambda;
        
    end
    
    
end

end


%%

function [dualityGap] = getDualityGap(Z,param,ActiveSet,inputData,obj)
        H = gradient(Z,inputData,param); % gradient
        temp = -1;
 if param.PSD
    for i = 1:length(ActiveSet.I)
        currTemp = PowerIteration(-H(ActiveSet.I{i}, ActiveSet.I{i}),param);
       
        if currTemp>temp
            temp = currTemp;
        end
        
        
    end
     
 else
     
    for i = 1:length(ActiveSet.I)
        currTemp = norm(H(ActiveSet.I{i}, ActiveSet.J{i}));
        if currTemp>temp
            temp = currTemp;
        end
    end
 end

switch param.f
    case 1 % prox
       
        kappa = min(1, param.lambda / temp) * (Z-inputData.Y);
        dualityGap = obj + sum(sum( inputData.Y .* kappa )) + 1 / 2 * sum(sum( kappa .* kappa ));
    case 2 % multitask
        
        kappa = min(1, param.lambda / temp) * (inputData.X*Z - inputData.Y);
        dualityGap = obj + sum(sum( inputData.Y .* kappa )) + 1 / 2 * sum(sum( kappa .* kappa ));
        
  
    case 3 % quadratic regression
        
        
        kappa = min(1, param.lambda / temp) * (diag(inputData.X*Z*inputData.X')-inputData.Y);
        dualityGap = obj + sum( inputData.Y .* kappa ) + 1 / 2 * sum( kappa .* kappa );

    case 4 % bilinear regression
        
        kappa = min(1, param.lambda / temp) * (inputData.X1*Z*inputData.X2-inputData.Y);
        dualityGap = obj + sum(sum( inputData.Y .* kappa )) + 1 / 2 * sum(sum( kappa .* kappa ));

end

end



%%

function [u v] = truncatedPowerSeq(A,param)

v = randn(size(A,2),1);v = v/norm(v);
if param.PSD
    u = v;
else
    u = randn(size(A,1),1);u = u/norm(u);
end

if A ==0
    u = projectL0(u,param.k);v = projectL0(v,param.q);
elseif param.PSD
     u = conGradU02(A ,param.powerIter,param.k); v = u;
else
    for i = 1:param.powerIter
        v = A'*u;
        v = projectL0(v,param.q);
        v = v/norm(v);
        
        u = A*v;
        u = projectL0(u,param.k);
        u = u/norm(u);

    end
end


end
    

%%


function [lambda u v] = PowerIteration(A,param)

v = randn(size(A,2),1);v = v/norm(v);
if param.PSD
    u = v;
else
    u = randn(size(A,1),1);u = u/norm(u);
end
if A ==0
    u = projectL0(u,param.k);v = projectL0(v,param.q);
elseif param.PSD
    for i = 1:param.powerIter
        u = A'*u;
        u = u/norm(u);
    end
    v = u;
else
    for i = 1:param.powerIter
        v = A'*u;
        v = v/norm(v);
        u = A*v;
        u = u/norm(u);
   
    end
end

lambda = sum(u.*(A*v));
end
    

%%


function y = projectL0(x,k)
 [z ix] = sort(abs(x),'descend');
 y = zeros(size(x));
 y(ix(1:k)) = sparse(x(ix(1:k)));

end




%%

function [S u v s grad tracenorm] = ista(X,Y, param,Start)


if param.f < 4
    L = 2*norm(X'*X);
else 
    L = 2*norm(X{1})^2*norm(X{2})^2 ;
end
if param.f == 3
        L = L^2;
end
S = Start;



obj = zeros(1,param.innerLoopIter);

for iter = 1:param.innerLoopIter
    
    if param.f == 2
        grad = X'*(X*S-Y);
    elseif param.f == 3
        grad = X'*diag(diag(X*S*X')-Y)*X;
    elseif param.f == 4
        grad = X{1}'*(X{1}*S*X{2}-Y)*X{2}';
    end
        
       
        S = S - 1/L*grad;

        if param.PSD
         [S tracenorm u s] = ShrinkPSD(S, param.lambda  / L ); v = u;
        else
         [S tracenorm u v s] = Shrink(S,param.lambda / L);
        end    
   %     fprintf('loss = %2.3e, pen = %2.3e, obj = %2.3e\n', .5*norm((X{1}*S*X{2}-Y), 'fro'), tracenorm,.5*norm((X{1}*S*X{2}-Y),'fro') + param.lambda*tracenorm)
end

end


%%

function [S u v s grad tracenorm] = fista(X,Y, param,Start)

if param.f < 4
    L = 2*norm(X'*X);
else 
    L = 2*norm(X{1})^2*norm(X{2})^2 ;
end
S = Start;
Sx = Start;
tk = 1;
tkp1 = tk;

 if param.f == 3
        L = L^2;
 end


for iter = 1:param.innerLoopIter
    
    if param.f == 2
        grad = X'*(X*S-Y);
    elseif param.f ==3
        grad = X'*diag(diag(X*S*X')-Y)*X;
    elseif param.f == 4
        grad = X{1}'*(X{1}*S*X{2}-Y)*X{2}';
    end
        
        SxOld = Sx;     
        Sx = S - 1/L*grad;

        if param.PSD
         [Sx tracenorm u s] = ShrinkPSD(Sx, param.lambda  / L ); v = u;
        else
          [Sx tracenorm u v s] = Shrink(Sx,param.lambda / L);
        end 
        tk = tkp1;
        tkp1 = .5*(1+sqrt(1+4*tk^2));
        
        S = Sx + (tk-1)/tkp1*(Sx-SxOld);
   
end

end

%%


function H = gradient(Z,inputData,param)

switch param.f
    case 1 % prox
        H = Z - inputData.Y;
    case 2 % multitask
        H = inputData.X'*(inputData.X*Z - inputData.Y);
    case 3 % quadratic regression
        H = inputData.X'*diag(diag(inputData.X*Z*inputData.X')-inputData.Y)*inputData.X;
    case 4 %  
        H = inputData.X1'*(inputData.X1*Z*inputData.X2 - inputData.Y)*inputData.X2';
end

end


%%
function startingZ = setDefaultStartingZ(inputData,param)
if param.f == 1
    startingZ = zeros(size(inputData.Y));
elseif param.f == 2
    startingZ = zeros(size(inputData.X,2), size(inputData.Y,2));
elseif param.f == 3
    startingZ = zeros(size(inputData.X,2));
elseif param.f == 4
    startingZ = zeros(size(inputData.X1,2), size(inputData.X2,1));
end
end
%%
function [Sh tracenorm u s] = ShrinkPSD(S,tau)
    S = .5*(S+S');
    [u s] = eig(S);
    [ds o] = sort(diag(s),'descend');
    u = u(:,o);
    s = s(o,o);
    r = sum(ds>tau);
    Sh = u(:,1:r)*diag(ds(1:r)-tau)*u(:,1:r)';
    tracenorm = sum(ds(1:r)-tau);
    u = u(:,1:r);
    s = s(1:r,1:r) - tau*eye(r);
    s = diag(s);
end


%%

function [Sh tracenorm u v ds] = Shrink(S,tau)

[u s v] = svd(S);

ds = diag(s);
r = sum(ds>tau);
ds = diag(ds(1:r)-tau);
Sh = u(:,1:r)*ds*v(:,1:r)';
tracenorm = sum(ds(1:r));
u = u(:,1:r); v = v(:,1:r);
ds = diag(ds);
end


%%

function x = conGradU02(A,nit,k)
% also named GPower or TPower
x = randn(size(A,1),1);

    if A ==0
        x = projectL0(x,k);
    else

        for i=1:nit   
                x = A*x;
                x = projectL0(x,k);
                x = x/norm(x);
        end
    end

end





%%

function param = setParamDefaultValues(param)
%% param contains
%
% param.f : the function to minimize. (1) stands for L2 distance squared
% .5*|Z-Y|^2, (2) stands for multitask .5*|XZ - Y|_F^2 and (3) for
% quadratic regression
%
% param.k param.q : blocks size
% param.PSD : true if the ouput is desired to be PSD, false else (default)
% param.nbMainLoop : nb of runs of "Solve P_S", default = 5
% param.powerIter : nb runs power iteration, default = 500;  nit
% param.stPtPowerIter : nb of different starting points to try, default =
% 100
% param.lambda : lagrange multiplier
% param.epsStop : add a component to the working set when  varIJ >
% lambda*(1-param.epsStop), default .001
% param.innerLoopIter : iterations of the prox-grad loop P_S default 50


if ~isfield(param,'q')
    param.q = param.k;
end


if ~isfield(param,'PSD')
    param.PSD = false;
end

if ~isfield(param,'nbMainLoop')
    param.nbMainLoop = 100;
end


if ~isfield(param,'powerIter')
    param.powerIter = 100;
end
if ~isfield(param,'stPtPowerIter')
    param.stPtPowerIter = 100;
end

if ~isfield(param,'epsStop')
    param.epsStop = .1;
end

if ~isfield(param,'innerLoopIter')
    param.innerLoopIter = 100;
end

if ~isfield(param,'niterPS')
    param.niterPS =  200;
end

if ~isfield(param,'PSdualityEpsilon')
    param.PSdualityEpsilon = 1e-3;
end

end