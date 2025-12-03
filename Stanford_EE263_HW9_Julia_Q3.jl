using LinearAlgebra, Plots

# Question 3

include("readclassjson.jl")
data = readclassjson("term_by_doc.json")
A = data["A"]
term = data["term"]
document = data["document"]
m = data["m"]
n = data["n"] 

# Part a
norms = [norm(col) for col in eachcol(A)]  
A_tilde = A ./ reshape(norms, 1, :)
U, S, V = svd(A_tilde)

# Plot singular values
plot(S, xlabel="Index",
        ylabel="Singular value",
        title="Singular values of A w/normal columns",
        label="")
savefig("hw9_q3a.png")

# Part b
q = zeros(m)
q[53] = 1
c = A_tilde' * q
p = sortperm(c, rev=true)
println("Top 5 search results indices, full matrix:  ", p[1:5])

# Part c
ranks = [32, 16, 8, 4]
for r in ranks
    temp = svd(A_tilde)
    A_hat = temp.U[:, 1:r] * Diagonal(temp.S[1:r]) * temp.Vt[1:r, :]
    c_hat = A_hat' * q
    p_hat = sortperm(c_hat, rev=true)
    println("Top 5 search results indices, low rank matrix s.t. r = $r:  ", p_hat[1:5])
end