# # Define    C_i = a_{2i} a_{2i}^T - a_{2i-1} a_{2i-1}^T (note symmetric - can use to speed up code)
# # and         w_i = b_{2i} - b_{2i-1}.
# #
# # (Here a_j is jth COLUMN of matrix A.)
# function transform_data(A, b)
#     (d,m) = size(A);
#     n = convert(Int64, m/2);
#
#     # C = zeros(d,d,n);
#     # CT = zeros(n,d,d);
#     #w = zeros(n);
#     grad_coeff = zeros(n,d,d);
#
#     for i=1:n
#         # a_even = A[ :, 2*i  ];
#         # a_odd =A[ :, 2*i-1 ];
#         # C[:,:,i] = a_even * a_even' - a_odd* a_odd'; # maybe can speed this up?
#
#         for k=1:d
#             grad_coeff[i,k,:] = A[k,2*i] * A[:,2*i] - A[k,2*i-1] * A[:,2*i-1];
#         end
#         #w[i] = b[2*i] - b[2*i-1];
#     end
#
#     # # transpose is what we really need for computing subgradients:
#     # for k=1:d
#     #     CT[:,k,:] = C[:,k,:]' ;
#     # end
#
#     # return CT
#     return grad_coeff
# end

# Compute the residuals r, objective value, and subgradient V for the covariance estimation objective:
# F(X) = (1/n) * ∑_{i=1}^n | trace( X X^T C_i ) - w_i |
#         = (1/n) * ∑_{i=1}^{n} |   || a_{2i}^T * X ||^2 -  || a_{2i-1}^T * X ||^2 - b_{2i} + b_{2i-1}   |
function cov_residuals!(res, A, b, XT, n)
    for i=1:n
        res[i] = sum(abs2, XT * A[:,2*i] )  - sum( abs2, XT * A[:,2*i-1] ) - b[2*i] + b[2*i-1];
    end
end

function cov_objective(res,n)
    (1/n) * sum(abs, res);
end

# Compute a subgradient (d x r matrix) V at X, using the chain rule.
# Write objective as:
# F(X) = (1/n) * || G(X) ||_1,      where G_i(X) = || X^T a_{2i} ||^2 - || X^T a_{2i-1} ||^2 - const
# Subgradient is then:
# V = ∇G(X)^T * [ (1/n) * ∂ || ⋅ ||_1 ( G(x) ) ] ,      where ∇G_i(X) = 2 * a_{2i} * a_{2i}^T *X - 2 * a_{2i-1} * a_{2i-1}^T *X
# (see matrix cookbook: derivative of Tr(BXX^T) is BX + B^TX)
# signs is subgradient of 1-norm at G(X)
function subgrad!(V, A, signs, X, n)
    scale!(V,0.0);
    for i=1:n
        gradGi = A[:,2*i] * ( A[:,2*i].' * X) - A[:,2*i-1]  * ( A[:,2*i-1].' * X);
        BLAS.axpy!(signs[i], gradGi, V);
    end
    scale!(V,2/n);
end
