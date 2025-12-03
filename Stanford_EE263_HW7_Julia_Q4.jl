using LinearAlgebra, Plots, Statistics

#Question 4.)

include("readclassjson.jl")
data = readclassjson("linexp_data.json")

Y = data["Y"]
n, N = size(Y)

#Choose b
b = sum(Y, dims=2) / N
tildeY = Y .- b  

#Choose m
U, Sigma, V = svd(tildeY; full=false) 
threshold = 0.95
cumsum_sv = cumsum(Sigma) / sum(Sigma)
m = findfirst(x -> x >= threshold, cumsum_sv) 

# Truncate SVD
Uhat = U[:, 1:m]
Sigmahat = Diagonal(Sigma[1:m])
Vhat = V[:, 1:m]

#Choose X and A
X = sqrt(N) * Vhat'
A = (1/sqrt(N)) * Uhat * Sigmahat

#Compute + Sort + Plot Errors
errors = [norm(Y[:,i] - A*X[:,i] - b) for i in 1:N]
sorted_errors = sort(errors, rev=true)
plot(sorted_errors, xlabel="Sorted Index", ylabel="Norm of Error Vector", 
     title="Distribution of Explanation Errors", legend=false)
savefig("hw7_q4b.png")

#Verify X vector properties
mean_X = sum(X, dims=2) / N
cov_X = (X * X') / N

println("Mean of x vectors (should be 0):")
println(mean_X)
println("m = ", m)
println("\nCovariance of x vectors (should be I_m):")
println(cov_X)