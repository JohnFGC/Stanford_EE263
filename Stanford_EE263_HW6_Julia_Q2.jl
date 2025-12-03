using LinearAlgebra, Plots
using Statistics

#Question 2.)
#Part B

include("readclassjson.jl")
data = readclassjson("gauss_fit_data.json")

t = data["t"]
y = data["y"]
N = data["N"]

plot(t, y, label = "Error Points", seriestype=:scatter)
xlabel!("t")
ylabel!("y")
title!("y plot")
savefig("hw6_q2b_i.png")

function linearize(p)
    a, mu, sigma = p
    A = zeros(N, 3)
    b = zeros(N)
    r = zeros(N)
    for i in 1:N
        ti = t[i]
        fi = f(ti, a, mu, sigma)
        r[i] = fi - y[i]
        A[i, 1] = exp(-(ti - mu)^2 / sigma^2)              
        A[i, 2] = fi * (2 * (ti - mu) / sigma^2)           
        A[i, 3] = fi * (2 * (ti - mu)^2 / sigma^3)
        b[i] = fi * (2 * ti * (ti - mu) / sigma^2) + y[i]
    end
    return A, b, r
end

function gauss_newton(p_start)
    maxIterations = 50
    tolerance = 1e-8
    p = copy(p_start)
    rms_errors = Float64[]
    for k in 1:maxIterations
        A, b, r = linearize(p)
        p_new = A \ b
        push!(rms_errors, sqrt(mean(r.^2)))
        if norm(p_new - p) < tolerance
            println("Converged after $k iterations")
            return p_new, rms_errors
        end
        p = p_new
    end
    println("Didn't converge")
    return p, rms_errors
end

p_good = [15, 50, 20]
p_diff = [13, 55, 25]
p_bad = [8, 0, 100]
print(p_good, p_diff, p_bad)
guesses = Dict(
    "Good guess" => p_good,
    "Different guess" => p_diff,
    "Bad guess" => p_bad
)

f(t, a, mu, sigma) = a * exp(-(t - mu)^2 / sigma^2)

for (label, p_start) in guesses
    p_final, rms_errors = gauss_newton(p_start)

    plot(1:length(rms_errors), rms_errors, xlabel="Iteration", ylabel="RMS Error", title="RMS Error vs Iteration ($label)", lw=2, marker=:o)
    savefig("hw6_q2b_ii_$(label[1:3]).png")

    fitted_gaussian = [f(tt, p_final...) for tt in t]
    scatter(t, y, label="Data", xlabel="t", ylabel="y", title="Gaussian Fit ($label)", legend=:top)
    plot!(t, fitted_gaussian, lw=2, label="Fitted Gaussian")
    savefig("hw6_q2b_iii_$(label[1:3]).png")
end