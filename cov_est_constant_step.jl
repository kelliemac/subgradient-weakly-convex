workspace()

using PyPlot
using Distributions
include("cov_est_func.jl");
include("solve_cov_est.jl");

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax  = 1000;
m  = 3000;      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 1;           # rank

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
srand(123);
A  = randn(d,m); # columns are measurement vectors, entries are iid N(0,1)

Xtrue = randn(d,r);
XTtrue = Xtrue.';

distrib = Bernoulli(0.1);
z = rand(distrib, m);
distrib2 = Normal(0.0, 10.0);
ζ = rand(distrib2, m);

b = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow
# b = (1-z) .* b + z .* abs(ζ);

#--------------------------------------------------------------------
#  Generate initial point
#--------------------------------------------------------------------
init_radius = 0.01 * vecnorm(Xtrue);  # should be like gamma * mu / rho. make sure to scale with dimension
pert = randn(d,r);
X0 = Xtrue + (init_radius  / vecnorm(pert) )* pert ;     # so that || X0- Xtrue ||_F = init_radius.

#--------------------------------------------------------------------
#   Set up the plot
#--------------------------------------------------------------------
clf();
xlabel(L"Iteration $k$");
ylabel(L"$dist \, (X_k,\mathcal{X}^*) \; / \;  || \bar X ||_F$");

# αVals =[1.0, 1/3, 1/9];
αVals = [1/3];

for α in αVals
        #--------------------------------------------------------------------
        #  Solve and add to plot
        #--------------------------------------------------------------------
         (Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Constant", stepSize=α);      # no noise, Polyak step
        semilogy(err_hist);
end

savefig("constant_step_error.pdf");
