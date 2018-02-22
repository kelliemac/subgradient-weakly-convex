# Function to solve covariance estimation problems
# (for use on synthetic problems for the weakly convex subgradient project)
#
# Kellie J. MacPhee
#
# Input: A = m x d measurement matrix (rows a1 ... an are the measurement vectors)
#           b = m-dim vector of observations
#         X0 = d x r matrix initilization
#       gopt = optimal value of problem (for Polyak step)
#       step = Polyak, constant, .... (right now only supports Polyak)
#       stepSize = size for constant step

# Returns: optimal X, error history,

include("cov_est_func.jl");

function solve_cov_est( A, b, X0, Xtrue; iter_max::Int=500, Tol::Float64=1e-15,
    OptVal::Float64=0.0, step::String="Polyak",
    stepSize::Float64=1.0,
    μ::Float64=0.0, L::Float64=1e10, ρ::Float64=1e10)

    # Take in problem data:
    (d, m) = size(A);
    n = convert(Int64, m/2);
    r = size(X0,2);
    normXtrue = vecnorm(Xtrue);

    # Initialize and pre-allocate space:
    k = 0;
    Xk = copy(X0);           # current iterate
    XT = Xk.';

    res = zeros(n,1);        # residuals, use for objective value
    cov_residuals!(res,A,b,XT,n);

    gk = cov_objective(res,n);  # objective value
    Vk = zeros(d,r);    # current subgradient
    gradGi = zeros(d,r);     # to help with computing subgradient
    subgrad!(Vk, gradGi, A, res, Xk, n);

    # If constant step size chosen, set now.
    if step=="Constant"
        α = stepSize;
    elseif step=="Decay"
        δ=0.2;
        κ=μ/L;
        q=sqrt(1-(1-δ)*κ^2);
        α0 = (δ*μ^2/(ρ*L));
    end

    # Allocate space to keep track of objective values and relative errors:
    obj_hist = fill(NaN, iterMax);
    err = 0.0;
    err_hist =  fill(NaN, iterMax);

    while k < iter_max
        # If subgradient is zero, done.
        if norm(Vk) <= 1e-14
            break
        end

        # Otherwise, update Xk (if step not Polyak, assume constant):
        if step=="Polyak"
            αk = (gk - OptVal) / sum(abs2, Vk);
        elseif step=="Constant"
            αk = α / vecnorm(Vk);
        elseif step=="Decay"
            αk = ( α0 / vecnorm(Vk) ) * q^k;
        end
        BLAS.axpy!( -αk, Vk, Xk);

        # Compute new objective value gk and subgradient Vk:
        transpose!(XT,Xk);
        cov_residuals!(res,A,b,XT,n);
        gk = cov_objective(res,n);
        subgrad!(Vk, gradGi, A, res, Xk, n)
        k = k + 1;  # number of rounds completed

        # Record objective value and relative error:
        obj_hist[k] = gk;
        err = normXtrue^2 + sum(abs2, Xk) - 2 * sum(svdvals(XT * Xtrue));
        err = sqrt(err)/normXtrue;
        err_hist[k] = err;

        # Print output to the console:
        @printf("iter %3d, obj %1.2e, err %1.2e, step %1.2e\n", k, gk, err, αk);

        if err <= Tol
            break
        end

    end

    return (Xk, obj_hist, err_hist)

end
