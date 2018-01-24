using PyPlot

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax    = 500;
tol        = 1e-10;
normalizeA = false;
init_sol   = false;

m  = 2000;      # row number
n  = 500;       # col number
σ  = 0.0;      # noise level added to b
exp_num = 1;

#--------------------------------------------------------------------
#   Objective funtion
#--------------------------------------------------------------------
# include("../func.jl");

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
for i=1:exp_num

srand(123);
A  = randn(m,n);
if normalizeA
    nA = map!(sqrt,sumabs2(A,2));
    broadcast!(/,A,A,nA);
end
xt = randn(n);
b  = map!((x) -> x^2 + σ*rand(), A*xt);
AT = A.';

ρ=norm(A)^2/m;


#--------------------------------------------------------------------
#  Initialization
#--------------------------------------------------------------------





#--------------------------------------------------------------------
#  Apply Solver
#--------------------------------------------------------------------

x = copy(xin);

#err_his = solve_sign_retrieval(x, xt, A, AT, b);

err_his = solve_sign_retrieval_decay(x, xt, A, AT, b, ρ, 0.06, 10000)

#--------------------------------------------------------------------
#  Plot
#--------------------------------------------------------------------

#z=linspace(0,length(err_his)-1,length(err_his));
#semilogy(z,err_his);
clf();
xlabel(L"Iteration $k$");
ylabel(L"$\|x_k-\bar x\|$");
semilogy(err_his);
savefig("error.pdf");


 m=convert(Int64,ceil(m+0.25*n));
end
