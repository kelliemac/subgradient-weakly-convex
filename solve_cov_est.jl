# Function to solve covariance estimation problems
# (for use on synthetic problems for the weakly convex subgradient project)
#
# Kellie J. MacPhee
#
# Input: A = n x d measurement matrix (rows a1 ... an are the measurement vectors)
#        b = n - dim vector of observations
#        X0 = d x r matrix initilization
#


# Returns: optimal X

function solve_cov_est( A, b, X0; OptVal = 0.0, iter_max=500, step="Polyak")

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

    while k < iter_max

        # Compute objective value gk and subgradient Vk:
        (gk, Vk) = cov_est_func( C, e, Xk);

        # If subgradient is zero, exit.
        if Vk==0
            return Xk
        end

        # Otherwise, take a step:
        if step="Polyak"
            αk = (gk-gopt)/vecnorm(Vk)^2;
        end
        Xk = Xk - α * Vk;
        k = k + 1;

    end

(gk, Vk) = cov_est_func( C, e, Xk);
return (Xk, gk, Vk)

end
