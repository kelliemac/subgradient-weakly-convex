# Compute the residuals inside the covariance estimation objective:
function cov_residuals!(res, A, b, XT, n)
    for i=1:n
        res[i] = sum(abs2, XT * A[:,2*i] )  - sum( abs2, XT * A[:,2*i-1] ) - b[2*i] + b[2*i-1];
    end
end

# Compute the covariance estimation objective:
function cov_objective(res,n)
    (1/n) * sum(abs, res);
end

# Compute a subgradient (d x r matrix) V at the point X, using the chain rule.
# F(X) = (1/n) * || G(X) ||_1,      where G_i(X) = < XX^T , a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T > - (b_{2i} - b_{2i-1})
# so subgradient is
# V = ∇G(X)^T * [ (1/n) * ∂ || ⋅ ||_1 ( G(x) ) ]
# where ∇G_i(X) = 2 ( a_{2i} * a_{2i}^T * X -  a_{2i-1} * a_{2i-1}^T * X )
# (See matrix cookbook: derivative of Tr(BXX^T) is BX + B^TX )
function subgrad!(V, gradGi, A, res, X, n)
    scale!(V,0.0);      # reset subgradient to 0
    for i=1:n
        # # add even term
        # BLAS.gemm!('Y', 'N', sign(res[i]), A[:,2*i], X, 0.0, Vtemp);        # would be faster without transpose
        # BLAS.gemm!('N', 'N', 1.0, A[:,2*i], Vtemp, 1.0, V);
        #
        # # subtract odd term
        # BLAS.gemm!('Y', 'N', sign(res[i]), A[:,2*i-1], X, 0.0, Vtemp);        # would be faster without transpose
        # BLAS.gemm!('N', 'N', -1.0, A[:,2*i-1], Vtemp, 1.0, V);

        gradGi[:,:] = A[:,2*i] * ( A[:,2*i].' * X) - A[:,2*i-1]  * ( A[:,2*i-1].' * X);     # maybe could speed this up
        BLAS.axpy!(sign(res[i]), gradGi, V);
    end
    scale!(V,2/n);
end
