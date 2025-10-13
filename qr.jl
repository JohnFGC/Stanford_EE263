
using LinearAlgebra



# A is m x n
#
# returns
#      Q  m x m, orthogonal
#      R  m x n  upper triangular
#                R = [ R11  R12
#                      0     0  ]
#                R11   r x r,  upper triangular, invertible
#      P  n x n  permutation matrix
#
# such that A*P = Q*R
#
# This algorithm should be used when
#   - A is not full rank
#   - or you don't know the rank of A
#   - or you think A is close to a non-full rank matrix.
#
function pivotedqr(A)
    m,n = size(A)
    Z = qr(A, Val(true))
    Q = Matrix(collect(Z.Q))
    if m > n
        R = [Z.R; zeros(m - n, n)]
    else
        R = Z.R
    end
    P = Z.P
    return P, Q, R
end



# A is m x n
#
# returns
#      Q  m x m orthogonal
#      R  m x n  upper staircase
#
# such that A = Q*R
#
# This algorithm should be used when
# you know that A is skinny and full rank
#
function fullqr(A)
    m,n = size(A)
    Z = qr(A, Val(false))
    Q = Matrix(collect(Z.Q))
    if m > n
        R = [Z.R; zeros(m - n, n)]
    else
        R = Z.R
    end
    return Q, R
end

function testqr()
    # simplest case: A is skinny and full rank
    A = [1 1
         2 1
         3 1];
    m,n = size(A)
    Q, R = fullqr(A)
    @assert size(Q) == (m,m)
    @assert size(R) == (m,n)
   


    P, Q, R = pivotedqr(A)
    @assert size(Q) == (m,m)
    @assert size(R) == (m,n)
end
