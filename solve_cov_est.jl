#
# Polyak step for subgradient method on covariance estimation
#
# Inputs:
#       A: (d x m) matrix with measurement vectors as columns
#       b: m-vector of observations
#       X0: (d x r) matrix to initalize algorithm
#       Xtrue: (d x r) matrix that we want to converge to
#       OptVal: optimal function value (value at Xtrue)
#       IterMax
#       Tol: tolerance for exiting algorithm based on norm of subgradient
#
# Outputs:
#       Xk: final estimate of Xtrue upon exiting algorithm
#       obj_hist: vector of objective values at each stem
#       err_hist: vector of normalized distances to solution set at each iteration

include("cov_est_func.jl");

function solve_cov_est( A, b, X0, Xtrue; OptVal::Float64=0.0, iterMax::Int=500,
    Tol::Float64=1e-15)

    # Take in problem data:
    (d, m) = size(A);
    AT = A.';
    r = size(X0,2);
    sqnormXtrue = sum(abs2, Xtrue);
    XTtrue = Xtrue.';

    # for coupling the even and odd measurements
    if iseven(m)
        n = convert(Int64, m/2);
    else
        throw(ArgumentError("A must have an even number of columns"))
    end

    # Initialize and pre-allocate space for variables
    k = 1;
    Xk = copy(X0);           # current iterate

    # compute residuals
    S = AT * Xk;    # will reuse this!
    res = zeros(n,1);
    cov_residuals!(res,S,b);

    # compute objective value and subgradient
    gk = cov_objective(res);
    Vk = zeros(d,r);
    subgrad!(Vk, A, res, S);

    # Allocate space to keep track of method's progress:
    obj_hist = fill(NaN, iterMax);
    obj_hist[k] = gk;

    err_hist =  fill(NaN, iterMax);
    err = sqnormXtrue + sum(abs2, Xk) - 2 * sum(svdvals(XTtrue * Xk));  # small svd (d x d)
    err = sqrt( err / sqnormXtrue );
    err_hist[k] = err;

    @printf("Starting values: obj %1.2e, err %1.2e, step %1.2e\n", gk, err, α/vecnorm(Vk) );

    while k < iterMax
        # If subgradient is zero, done.
        if norm(Vk) <= Tol
            break
        end

        # Otherwise, update Xk
        αk = (gk-OptVal)/sum(abs2,Vk);
        BLAS.axpy!( -αk, Vk, Xk);

        # Update objective value and subgradient
        BLAS.gemm!('N', 'N', 1.0, AT, Xk, 0.0, S) # update S = AT * Xk
        cov_residuals!(res,S,b);
        gk = cov_objective(res);
        subgrad!(Vk, A, res, S)

        k = k + 1;

        # Record objective value and (normalized) distance to solution set
        obj_hist[k] = gk;
        err = sqnormXtrue + sum(abs2, Xk) - 2 * sum(svdvals(XTtrue * Xk));  # small svd (d x d)
        err = sqrt( err / sqnormXtrue );
        err_hist[k] = err;

        # Print current status to the console
        @printf("iter %3d, obj %1.2e, err %1.2e, step %1.2e\n", k, gk, err, αk);

    end

    # Done!
    return (Xk, obj_hist, err_hist)

end
