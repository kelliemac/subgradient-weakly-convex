workspace()

using PyPlot
include("cov_est_func.jl");
include("solve_cov_est.jl");

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax  = 1500;
m  = 5000;      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 2;           # rank
σ  = 0.0;      # noise level in observations

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
srand(123);
A  = randn(d,m); # columns are measurement vectors, entries are iid N(0,1)

Xtrue = randn(d,r);
XTtrue = Xtrue.';

b = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow
noise = randn(m,1);
b = b + σ*noise;

#--------------------------------------------------------------------
#  Generate initial point
#--------------------------------------------------------------------
init_radius = 1.0;  # should be like gamma * mu / rho. make sure to scale with dimension
pert = randn(d,r);
X0 = Xtrue + (init_radius  / vecnorm(pert) )* pert ;     # so that || X0- Xtrue ||_F = init_radius.

#--------------------------------------------------------------------
#   Set up the plot
#--------------------------------------------------------------------
clf();
xlabel(L"Iteration $k$");
ylabel(L"$ \min_{\Omega} \quad || \Omega X_k-\bar X ||^2 \; / \;  || \bar X ||^{2}$");
# title("Relative distance to solution set (Constant step)")

αVals =[1e-3, 5e-4, 1e-4, 5e-5]; #[1.0, 1/3, 1/9];

for α in αVals
        #--------------------------------------------------------------------
        #  Solve and add to plot
        #--------------------------------------------------------------------
         (Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Constant", stepSize=α);      # no noise, Polyak step
        semilogy(err_hist, label=(L"α="*string(α)));
end

legend();
savefig("constant_step_error.pdf");
