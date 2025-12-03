using LinearAlgebra

#Question 4.)

Ameas = [2.0  1.2 -1.0;
         0.4  2.0 -0.5;
        -0.5  0.9  1.0]

v1 = [0.7, 0.0, 0.7]
v2 = [0.3, 0.6, 0.7]
v3 = [0.6, 0.6, 0.3]

T = hcat(v1, v2, v3) 
invT = inv(T)
n = 3

C = zeros(n*n, n)
for i in 1:n
    e = zeros(n)
    e[i] = 1.0
    B_i = T * Diagonal(e) * invT                 
    C[:, i] = B_i[:]             
end

Ameas_tilde = Ameas[:]
lambda_opt = C \ Ameas_tilde
Ahat = T * Diagonal(lambda_opt) * invT
eigvals_Ahat, eigvecs_Ahat = eigen(Ahat)         
J = (1/(n*n)) * sum((Ameas - Ahat).^2)

println("Optimal Eigenvalues = ", lambda_opt)
println("Ahat = \n", Ahat)
println("Frobenius cost J = ", J)
println("Eigenvectors of Ahat = ", eigvecs_Ahat)
println("Eigenvalues of Ahat = ", eigvals_Ahat)
