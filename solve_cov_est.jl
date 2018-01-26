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


# Returns: optimal X

function solve_cov_est( A, b, X0, Xtrue; OptVal= 0.0, iter_max=500, step="Polyak")

    # Take in problem data:
    (m, d) = size(A);
    (d2, r) = size(X0);
    (C, e) = transform_data(A, b);

    # # Throw errors if input is unacceptable:
    # if d2 != d
    #     throw(DimensionMismatch("sizes of A and X0 do not match"))
    # elseif length(b) != n
    #     throw(DimensionMismatch("sizes of A and b do not match"))
    # end

    # Initialize:
    k = 0;
    Xk = copy(X0);
    (gk, Vk) = cov_est_func( C, e, Xk);

    # Allocate space to keep track of objective values and relative errors:
    obj_hist = fill(NaN, iterMax);
    err_hist =  fill(NaN, iterMax);

    while k < iter_max
        # If subgradient is zero, done.
        if Vk==0
            return Xk
        end

        # Otherwise, take a step:
        if step=="Polyak"
            αk = (gk - OptVal) / vecnorm(Vk)^2;
        end
        Xk = Xk - αk * Vk;

        # Compute new objective value gk and subgradient Vk:
        (gk, Vk) = cov_est_func( C, e, Xk);
        k = k + 1;

        # Record objective value and relative error:
        obj_hist[k] = gk;
        err_hist[k] = vecnorm(Xk-Xtrue)/vecnorm(Xtrue);

    end

return (Xk, obj_hist, err_hist)

end
