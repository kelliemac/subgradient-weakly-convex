# Compute the residuals inside the covariance estimation objective:
function cov_residuals!(res, S, b)
    for i=1:length(res)
        res[i] = sum(abs2, S[2*i,:])  - sum( abs2, S[2*i-1,:] ) - b[2*i] + b[2*i-1];
    end
end

# Compute the covariance estimation objective:
function cov_objective(res)
    sum(abs, res) / length(res);
end

# Compute a subgradient (d x r matrix) V at the point X, using the chain rule.
# F(X) = (1/n) * || G(X) ||_1,      where G_i(X) = < XX^T , a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T > - (b_{2i} - b_{2i-1})
# so subgradient is
# V = ∇G(X)^T * [ (1/n) * ∂ || ⋅ ||_1 ( G(x) ) ]
# where ∇G_i(X) = 2 ( a_{2i} * a_{2i}^T * X -  a_{2i-1} * a_{2i-1}^T * X )
# (See matrix cookbook: derivative of Tr(BXX^T) is BX + B^TX )
function subgrad!(V, A, res, S)
    n = length(res);
    scale!(V,0.0);      # reset subgradient to 0
    for i=1:n
        BLAS.ger!(sign(res[i]), A[:,2*i], S[2*i,:], V); # rank one update
        BLAS.ger!(-sign(res[i]), A[:,2*i-1], S[2*i-1,:], V);
    end
    scale!(V,2/n);
end
