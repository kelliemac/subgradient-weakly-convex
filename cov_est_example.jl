# Create and solve an instance of the covariance estimation problem, with objective function:
#
# F(X) =  (1/n) * ∑_{i=1}^{n} | < XX^T , a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T > - (b_{2i} - b_{2i-1}) |
#         = (1/n) * ∑_{i=1}^n | trace( X X^T C_i ) - w_i |
#
# where the measurement vectors a_1, ... , a_m are encoded as the rows
# of an (m x r) matrix A, and m=2n is even. With no measurement noise,
# b_j = < a_j a_j^T , XX^T > for j = 1, ... , m.
#
# The optimal XX^T is a rank r approximation of the covariance matrix, and X is (d x r).
#
# Kellie J. MacPhee

workspace()

using PyPlot
# using Distributions       # for Gaussian random vectors
#using Base.Profile
#using ProfileView

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax    = 500;

m  = 3000;      # number of measurements (need m>2d)
d  = 1000;       # ambient dimension
r = 2;           # rank
σ  = 0.0;      # noise level in observations

#--------------------------------------------------------------------
#   Objective funtion and subgradients, solver
#--------------------------------------------------------------------
include("cov_est_func.jl");
include("solve_cov_est.jl");

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
# distrib = MvNormal(d, 1.0);
srand(123);

# A = (rand(distrib, m)) ;  # columns are measurement vectors - gaussian
A  = randn(d,m); # columns are measurement vectors

Xtrue = rand(d,r);
XTtrue = Xtrue.';

# b = zeros(m,1);
b = mapslices( x -> sum( abs2, XTtrue * x ) , A, [1])';  # slow
noise = rand(m,1);      # change this so noise is gaussian, too? right now σ=0.0 so doesn't really matter
b = b + σ*noise;

#--------------------------------------------------------------------
#  Generate initial point
#--------------------------------------------------------------------
init_radius = 1.0/vecnorm(Xtrue);  # should be like gamma * mu / rho

pert = rand(d,r);       # make gaussian instead?
X0 = Xtrue + (init_radius  / vecnorm(pert) )* pert ;     # so that || X0- Xtrue ||_F = init_radius.

#--------------------------------------------------------------------
#  Apply Solver
#--------------------------------------------------------------------

(Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Polyak");      # no noise, Polyak step

# α = 0.1/norm(A)^2;
# (Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Constant", stepSize=α);      # no noise, constant step

# I don't actually know what  these parameters should be, so I can't make this converge,
# but you can fill in μ and L and ρ and run the decaying step, too:
# (Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Decay", μ=?? , L=?? , ρ=?? );

#--------------------------------------------------------------------
#  Plot convergence (distance to optimum and objective value)
#--------------------------------------------------------------------

clf();
xlabel(L"Iteration $k$");
ylabel(L"$\|X_k-\bar X\|_F / \|\bar X\|_F$");
title("Relative distance to optimum")
semilogy(err_hist);
savefig("error.pdf");

# clf();
# xlabel(L"Iteration $k$");
# ylabel(L"$F(X_k)$");
# title("Objective value")
# semilogy(obj_hist);
# savefig("objectives.pdf");
