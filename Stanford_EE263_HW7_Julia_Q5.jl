using LinearAlgebra, Plots, Statistics

#Question 5.)

include("readclassjson.jl")
data = readclassjson("low_rank_matrix_completion.json")

m = data["m"]
n = data["n"]
r = data["r"]
Atrue = data["Atrue"]
Aknown = data["Aknown"]
K = data["K"]
p = data["p"]

Ahat = fill(mean(Aknown), m, n)
for i in 1:p
    Ahat[K[i, 1], K[i, 2]] = Aknown[i]
end
error = []
for i in 1:300
    F = svd(Ahat)
    Atilde = F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    global Ahat = Atilde
    for j in 1:p
        global Ahat[K[j, 1], K[j, 2]] = Aknown[j]
    end
    push!(error, norm(Ahat - Atrue))
end

p_tradeoff = plot(1:300, error, xlabel="k", 
                                ylabel="Frobenius Norm of Difference", 
                                title="Convergence of Low-Rank Matrix Completion", 
                                label="")
savefig("hw7_q5b.png")
println(error[300])