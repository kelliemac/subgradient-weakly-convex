workspace()

using PyPlot

include("cov_est_func.jl");
include("solve_cov_est_constant_step.jl");

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iter_max  = 10;
m  = 3000;      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 1;           # rank
pfail = 0.1;        # probabilty of an outlier

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
srand(123);
A  = randn(d,m); # columns are measurement vectors, entries are iid N(0,1)

# Generating outliers - for vector case only right now
Inc=rand(m);
for j=1:m
  if Inc[j,1]<pfail
    Inc[j,1]=1
  else
    Inc[j,1]=0
  end
end
Outliers=10*abs(randn(m));

Xtrue = randn(d,r);
XTtrue = Xtrue.';

b_temp = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow
b = vec((ones(m,1)-Inc).*b_temp+Inc.*Outliers);

#--------------------------------------------------------------------
#  Generate initial point
#--------------------------------------------------------------------
init_radius = 1.0 * vecnorm(Xtrue);  # should be like gamma * mu / rho. make sure to scale with dimension
pert = randn(d,r);
X0 = Xtrue + (init_radius  / vecnorm(pert) )* pert ;     # so that || X0- Xtrue ||_F = init_radius.

#--------------------------------------------------------------------
#   Set up the plot
#--------------------------------------------------------------------
clf();
xlabel(L"Iteration $k$");
ylabel(L"$dist \, (X_k,\mathcal{X}^*) \; / \;  || \bar X ||_F$");

αVals =[1.0, 1/3, 1/9];

for α in αVals
        #--------------------------------------------------------------------
        #  Solve and add to plot
        #--------------------------------------------------------------------
         (Xest, obj_hist, err_hist) = solve_cov_est_constant_step( A, b, X0, Xtrue; iterMax=iter_max, stepSize=α);      # no noise, Polyak step
        semilogy(err_hist);
end

savefig("constant_step_error.pdf");
