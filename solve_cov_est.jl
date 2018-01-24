# Function to solve covariance estimation problems
# (for use on synthetic problems for the weakly convex subgradient project)
#
# Kellie J. MacPhee
# Jan 23, 2018

# Input: A = n x d measurement matrix (rows a1 ... an are the measurement vectors)
#        b = n - dim vector of observations
#        X0 = d x r matrix initilization
#


# Returns: optimal X

function solve_cov_est( A, b, X0,  ; gopt=0, iter_max=500, step="Polyak")

    (n,d) = size(A);
    (d2,r) = size(X0);

    # Throw errors if input is unacceptable:
    if d2 != d
        throw(DimensionMismatch("sizes of A and X0 do not match"))
    elseif length(b) != n
        throw(DimensionMismatch("sizes of A and b do not match"))
    end


    # Initialize:
    k = 0;
    Xk = copy(X0);
    gk = # function value at Xk


    while k < iter_max

        # get a subgradient
        vk =

        if vk=0
            return Xk
        end

        # take a step with the appropriate step size
        if step="Polyak"
            αk = (gk-gopt)/norm(vk)^2;
        end

        Xk = Xk - α * vk;

        # project Xk onto feasible region?



        k = k + 1;

















end


# Function to compute objective value of covariance estimation problems

function cov_est_objective( B, b, X)
    # Compute the
