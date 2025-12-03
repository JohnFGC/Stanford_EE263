using LinearAlgebra, Plots

#Question 3.)

include("readclassjson.jl")
data = readclassjson("missing.json")
K = data["known"]
y = data["y"]
n = 100

#Part C

K_c = setdiff(1:n, K)

G = zeros(n - 1, n)
for i in 1:n-1
    G[i, i] = -1
    G[i, i + 1] = 1
end

G_K = G[:, K]
G_K_c = G[:, K_c]

z_K_c = - inv(G_K_c' * G_K_c) * G_K_c' * G_K * y
z_opt = zeros(n)
z_opt[K] = y
z_opt[K_c] = z_K_c

scatter(1:n, z_opt, label="z", xlabel="i", ylabel="z(i)", title="Optimal z, first derivative only", legend=:top)
scatter!(K, y, label="Known y", color=:red)
savefig("hw6_q3c.png")


#Part D

H = zeros(n - 2, n)
for i in 1:n-2
    H[i, i] = 1
    H[i, i + 1] = -2
    H[i, i + 2] = 1
end

H_K = H[:, K]
H_K_c = H[:, K_c]

mu_values = 10 .^ range(-3, 3, length=200)
J1 = []
J2 = []

for mu in mu_values
    z_tradeoff_K_c = - inv(G_K_c' * G_K_c + mu * H_K_c' * H_K_c) * (G_K_c' * G_K + mu * H_K_c' * H_K)* y
    z_tradeoff_opt = zeros(n)
    z_tradeoff_opt[K] = y
    z_tradeoff_opt[K_c] = z_tradeoff_K_c
    push!(J1, norm(G*z_tradeoff_opt)^2)
    push!(J2, norm(H*z_tradeoff_opt)^2)
end

p_tradeoff = plot(J1, J2, marker=:circle,
                        xlabel="J1",
                        ylabel="J2",
                        title="Tradeoff Curve",
                        label="Tradeoff")
savefig("hw6_q3d.png")


#Part E

mu_values_to_plot = [5, 20, 100]
for mu in mu_values_to_plot
    
    z_tradeoff_K_c = - inv(G_K_c' * G_K_c + mu * H_K_c' * H_K_c) * (G_K_c' * G_K + mu * H_K_c' * H_K)* y
    z_tradeoff_opt = zeros(n)
    z_tradeoff_opt[K] = y
    z_tradeoff_opt[K_c] = z_tradeoff_K_c
    scatter(1:n, z_tradeoff_opt, label="z", xlabel="i", ylabel="z(i)", title="Optimal z, with second der, mu = $mu", legend=:top)
    scatter!(K, y, label="Known y", color=:red)
    savefig("hw6_q3e_mu=$mu.png")
end