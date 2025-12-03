using LinearAlgebra, Plots, Statistics

#Question 2.)

include("readclassjson.jl")
data = readclassjson("bs_det_data.json")


Y = data["Y"]        
s = data["s"]    
T = data["T"]
n = data["n"]
R = (1/T) * (Y * Y')

eig = eigen(R)
eigenvalues = eig.values
eigenvectors = eig.vectors
i_max = argmax(eigenvalues)
u = eigenvectors[:, i_max]
lambda_u = eigenvalues[i_max]

w = -u / sqrt(lambda_u)
println("Weight vector w:")
println(w)

s_hat = w' * Y
histogram(vec(s_hat), bins=60,
    xlabel="s_hat",
    ylabel="count",
    title="Histogram for s_hat")
savefig("hw8_q2b.png")

s_est = map(x -> x >= 0 ? 1 : -1, vec(s_hat))
err = mean(s_est .!= s)
println("\nError rate = ", err)