using LinearAlgebra, Plots, Statistics

#Question 4.)

include("readclassjson.jl")
data = readclassjson("lena_data.json")
X = data["X"] 

#Part a
U, S, V = svd(X)

# Plot singular values
plot(S, xlabel="Index",
        ylabel="Singular value",
        title="Singular values of X",
        label="")
savefig("hw8_q4a.png")

#Part b
k = 75
X_tilde = U[:, 1:k] * Diagonal(S[1:k]) * V[:, 1:k]'

# Frobenius norm relative error
rel_error = norm(X - X_tilde) / norm(X)
println("Relative Frobenius norm error: ", rel_error)

# Display low-rank approximation
heatmap(X_tilde, c=:grays, yflip=true, axis=false, aspect_ratio=:equal)