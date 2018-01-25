# Define    C_i = a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T,
# and         e_i = b_{2i} - b_{2i-1}.
function transform_data( A, b )
    (m,d) = size(A);
    n = convert(Int64, m/2); # will give an error if m is odd

    C = zeros(d,d,n);
    for i=1:n
        a_even = A[ 2*i, : ];
        a_odd =A[ 2*i-1, : ];
        C[:,:,i] = a_even * a_even' - a_odd* a_odd'; # maybe can speed this up using BLAS.gemm! ?
    end

    e = zeros(n,1);
    for i=1:n
        e[i] = b[2*i] - b[2*i-1];
    end

    return (C,e)
end

# Compute (objective value, subgradient) for the covariance estimation objective:
# F(X) = (1/n) * ∑_{i=1}^{n} | B_i ( XX^T - X̄X̄^T ) |
#         = (1/n) * ∑_{i=1}^n | trace( X X^T C_i ) - e_i |
function cov_est_func( C, e, X)
        n = length(e);
        (d,r) = size(X);

        XXT = X * X' ;

        # Compute objective value:
        residual = zeros(n,1);
        for i=1:n
            residual[i] = trace( XXT * C[:,:,i] ) - e[i]
        end
        obj = (1/n) * sum(abs, residual);

        # Compute a subgradient (d x r matrix), using the chain rule:
        outer_subgrad = (1/n) * sign(residual);

        SUBGRAD = zeros(size(X));
        for k=1:d
            for l=1:r
                temp = 2 * X[:,l]' * C[:,:,k] * outer_subgrad;
                SUBGRAD[k,l] = temp[1];
            end
        end

        return (obj, SUBGRAD)
end
