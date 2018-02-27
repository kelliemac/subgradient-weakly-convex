workspace()

using PyPlot
include("cov_est_func.jl");
include("solve_cov_est.jl");

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax  = 500;
mVals = [5000];
# mVals  = [5000, 8000, 11000, 14000, 17000, 20000];      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 3;           # rank

#--------------------------------------------------------------------
#   Set up the plot
#--------------------------------------------------------------------
clf();
xlabel(L"Iteration $k$");
ylabel(L"$dist \, (X_k,\mathcal{X}^*) \; / \;  || \bar X ||_F$");

for m in mVals
        #--------------------------------------------------------------------
        #   Generate Data
        #--------------------------------------------------------------------
        srand(123);
        A  = randn(d,m); # columns are measurement vectors, entries are iid N(0,1)

        Xtrue = randn(d,r);
        XTtrue = Xtrue.';

        b = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow

        #--------------------------------------------------------------------
        #  Generate initial point
        #--------------------------------------------------------------------
        init_radius = 1.0 * vecnorm(Xtrue);
        pert = randn(d,r);
        X0 = Xtrue + (init_radius  / vecnorm(pert) )* pert ;     # so that || X0- Xtrue ||_F/ ||Xtrue||_F = init_radius.

        #--------------------------------------------------------------------
        #  Solve and add to plot
        #--------------------------------------------------------------------
        (Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iterMax=iter_max);
        semilogy(err_hist);

end

ylim(ymin=1e-7);
savefig("polyak_error.pdf");
