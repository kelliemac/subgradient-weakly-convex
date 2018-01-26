# Create and solve an instance of the covariance estimation problem, with objective function:
#
# F(X) =  (1/n) * ∑_{i=1}^{n} | < XX^T , a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T > - (b_{2i} - b_{2i-1}) |
#         = (1/n) * ∑_{i=1}^n | trace( X X^T C_i ) - e_i |
#
# where the measurement vectors a_1, ... , a_m are encoded as the rows
# of an (m x r) matrix A, and m=2n is even. With no measurement noise,
# b_j = < a_j a_j^T , XX^T > for j = 1, ... , m.
#
# The optimal XX^T is a rank r approximation of the covariance matrix, and X is (d x r).
#
# Kellie J. MacPhee

using PyPlot
using Distributions

#--------------------------------------------------------------------
#   Parameters
#--------------------------------------------------------------------

iterMax    = 500;

m  = 200;      # number of measurements
d  = 100;       # ambient dimension
r = 10;           # rank
σ  = 0.0;      # noise level in observations

init_radius = 0.01;     # normalized distance between initial estimate and true solution

#--------------------------------------------------------------------
#   Objective funtion and subgradients, solver
#--------------------------------------------------------------------
include("cov_est_func.jl");
include("solve_cov_est.jl")

#--------------------------------------------------------------------
#   Generate Data
#--------------------------------------------------------------------
distrib = MvNormal(d, 1.0);
srand(123);

A = (rand(distrib, m))' ;
#A  = randn(m,d);  # change this so measurement vectors (rows) are gaussian

Xtrue = rand(d,r);

b = zeros(m,1);
for i=1:m
    b[i] = (A[i,:]' * Xtrue * Xtrue' * A[i,:])[1];
end
noise = rand(m,1);      # change this so noise is gaussian, too? right now σ=0.0 so doesn't really matter
b = b + σ*noise;

#--------------------------------------------------------------------
#  Initialization
#--------------------------------------------------------------------

pert = rand(d,r);       # make gaussian instead?

X0 = Xtrue + init_radius * vecnorm(Xtrue) * pert ./ vecnorm(pert) ;     # so that || X0- Xtrue ||_F / || X_true ||_F = init_radius

#--------------------------------------------------------------------
#  Apply Solver
#--------------------------------------------------------------------

(Xest, obj_hist, err_hist) = solve_cov_est( A, b, X0, Xtrue; OptVal=0.0, iter_max=iterMax, step="Polyak");       # no noise, Polyak step


#--------------------------------------------------------------------
#  Plot convergence (distance to optimum and objective value)
#--------------------------------------------------------------------

clf();
xlabel(L"Iteration $k$");
ylabel(L"$\|X_k-\bar X\|_F / \|\bar X\|_F$");
title("Relative distance to optimum")
plot(err_hist);
savefig("error.pdf");

clf();
xlabel(L"Iteration $k$");
ylabel(L"$F(X_k)$");
title("Objective value")
semilogy(obj_hist);
savefig("objectives.pdf");
