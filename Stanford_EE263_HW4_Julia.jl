using LinearAlgebra
using Plots

include("readclassjson.jl")
data = readclassjson("curve_smoothing.json")

n = data["n"]
f = data["f"]

# Matrices
A = (1/sqrt(n)) * I
b = (1/sqrt(n)) * f
P = zeros(Float64, n-2, n)
for i in 1:n-2
    P[i, i]   = n^2
    P[i, i+1] = -2*n^2
    P[i, i+2] = n^2
end
P = (1/sqrt(n-2)) * P

# Tradeoff curve
mu_values = 10 .^ range(-6, 0, length=200)
d = []
c = []

for mu in mu_values
    g = inv(A' * A + mu * P' * P) * (A' * b)
    push!(d, norm(A*g - b)^2)
    push!(c, norm(P*g)^2)
end

p_tradeoff = plot(c, d, marker=:circle,
                        xlabel="c",
                        ylabel="d",
                        title="Tradeoff Curve",
                        label="Tradeoff")

# Smoothed curve
t = range(0, 1, length=n)
mu_values = [0.0, 1e-5, 1e-4, 1e-3, 1.0]
p_smooth = []

for mu in mu_values
    g = inv(A' * A + mu * P' * P) * (A' * b)
    
    p = plot(t, f, label="f", lw=2, linestyle=:dot)
    plot!(p, t, g, label="g (mu=$mu)", lw=2)
    xlabel!("t")
    ylabel!("g(t)")
    title!("Curve Smoothing for mu = $mu")
    
    push!(p_smooth, p)  # keep window alive
end

display(p_smooth[5])
println("Press Enter to exit...")
readline()
