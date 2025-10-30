using LinearAlgebra, Plots
using Statistics

#Question 1.)

#Reading Data
include("readclassjson.jl")
data = readclassjson("fleet_mod_data.json")

n = data["n"]
K = data["K"]
T = data["T"]
x_array = [data["X$i"]' for i=1:data["K"]]
y_array = [data["y$i"]' for i=1:data["K"]]


#Creating y vector and x portion of A matrix
x_matrix = vcat(x_array...)
y = vcat(y_array...)

#Creating identity portion of A matrix and combining
b_matrix = zeros(T*K, K)
for k in 1:K
    rows = (T*(k-1)+1):(T*k)
    b_matrix[rows, k] .= 1.0
end
A = [x_matrix b_matrix]

#Solving Least Squares
x_ls = A \ y
a = x_ls[1:n]
b = x_ls[n+1:end]

println()
println("Model parameter a = ", a)
println("Model parameter [b(1), ..., b(k)] = ", b)
println("Mean Square Error E = ", norm((A*x_ls) - y)^2 / (T*K))
println()

A_com = [x_matrix ones(T*K)]
x_com = A_com \ y
a_com = x_com[1:n]
b_com = x_com[n+1:end]
println("Common Offset Model parameter a = ", a_com)
println("Common Offset Model parameter b = ", b_com)
println("Common Offset Mean Square Error Ecom = ", norm((A_com*x_com) - y)^2 / (T*K))
println()



#Question 2.)

#Reading Data + Defining Variables
data2 = readclassjson("recursive.json")
x = data2["x"]
y = data2["y"]
w = data2["w"]

n = 10
k = 5
sigma = 2
q = 10
mu = 0.1
m = k*n

#Define A = BC
C= zeros(m, n)
for col in 1:n
    rows = (k*(col-1)+1):(k*col)
    C[rows, col] .= 1.0
end

r = [exp(-(j^2 / sigma^2)) for j in -q:q]

B = zeros(m, m)
for row in 1:m
    for col in 1:m
        r_idx = row - col
        if abs(r_idx) <= q
            B[row, col] = r[r_idx + (q + 1)]
        end
    end
end

A = B * C

#Part C
x_reg = inv((A' * A) + (mu * I(n))) * A' * y
plot(x, label = "x", lw = 2, linestyle=:dash)
plot!(x_reg, label = "x_reg", lw = 1)
xlabel!("i")
ylabel!("x(i)")
title!("x reg and x for Part C")
savefig("hw5_q2i")

#Part E
P = mu * I(n)
q = zeros(n)

plot(x, label = "x", lw = 2, linestyle=:dash)
for i in 1:m
    ai = A[i, :]
    yi = y[i]
    global P += ai * ai'
    global q += yi * ai
    xreg = P \ q
    if i == 18
        x_estimate = P \ q
        plot!(x_estimate, label = "Estimated x for i = 18", lw = 1)
    end
    if i == 30
        x_estimate = P \ q
        plot!(x_estimate, label = "Estimated x for i = 30", lw = 1)
    end
end
xlabel!("i")
ylabel!("x(i)")
title!("x estimates and x for Part E")
savefig("hw5_q2iii")

#Part F
for i in 1:20
    ai = A[i, :]
    yi = y[i]
    global P -= ai * ai'
    global q -= yi * ai
end

x_improved_est = P \ q
plot(x, label = "x", lw = 2, linestyle=:dash)
plot!(x_improved_est, label = "Estimated x w/out bad data", lw = 1)
xlabel!("i")
ylabel!("x(i)")
title!("Improved x estimate and x for Part F")
savefig("hw5_q2v")



#Question 3.)
G = [2 3;
     1 0;
     0 4;
     1 1;
     -1 2]

G_tilde = [-3 -1;
           -1  0;
            2 -3;
           -1 -3;
            1  2]

H = [-0.72  2.84 0.64 -0.4 0;
      0.32 -0.04 0.16 -0.6 0]
println()
println("HG = ", H*G)
println("HG_tilde = ", H*G_tilde)
println()



#Question 4.)

A = [-1 1 -1 -1 0 0 0;
     0 -1 0 0 -1 0 0;
     0 0 0 1 1 -1 0;
     0 0 1 0 0 1 -1]
s = [1 4 10 10]'
n = 7
f_simple = [5 4 0 0 0 10 20]'

f = -A' * inv(A * A') * s

println("f_optimal Mean Square Error = ", norm(f)^2 / n)
println("f_simple Mean Square Error Ecom = ", norm(f_simple)^2 / n)
