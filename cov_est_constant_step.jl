workspace()

using PyPlot

include("cov_est_func.jl");
include("solve_cov_est_constant_step.jl");

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iter_max  = 2000;
m  = 10000;      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 3;           # rank
pfail = 0.1;        # probabilty of an outlier
σ = 10.0;        # std dev of outliers

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
srand(123);
A  = randn(m,d).'; # columns are measurement vectors, entries are iid N(0,1)

# Generating outliers - for vector case only right now
Inc=rand(m,d);
for j=1:m
    for k=1:d
        if Inc[j,k]<pfail
            Inc[j,k]=1;
        else
            Inc[j,1]=0;
        end
    end
end
Outliers=σ*abs(randn(m));

Xtrue = randn(d,r);
XTtrue = Xtrue.';

b_temp = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow
b = (ones(m,d)-Inc) .* b_temp + Inc .* Outliers;
# b = b_temp;

# # FOR TESTING AGAINST SIGN RETRIEVAL: decouple even and odd terms
# Afull = zeros(d,2*m);
# for i=1:m
#     Afull[:,2*i] = A[:,i];
# end
#
# bfull = vec(zeros(2*m,1));
# for i=1:m
#     bfull[2*i] = b[i];
# end

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

# αVals = [1.0, 1/3, 1/9];
αVals = [1.0, 1/3, 1/9];

for α in αVals
        #--------------------------------------------------------------------
        #  Solve and add to plot
        #--------------------------------------------------------------------
         (Xest, obj_hist, err_hist) = solve_cov_est_constant_step( A, b, X0, Xtrue; iterMax=iter_max, stepSize=α); 
        semilogy(err_hist);
end

savefig("constant_step_error.pdf");
