using LinearAlgebra
using Plots

include("qr.jl")

A = [1 2 1
     1 -1 -2
    -2 1 3
     1 -1 -2
     1 1 0]

Q, R = fullqr(A)

println()
println("Rank of R is ", rank(R), ", so Q2 has ", rank(Q) - rank(R)," basis vectors")
println()

Q2 = Q[:, 3:5]
B = Q2'

println("B = ", B)
println()
println("BA = ", B * A)
println()
println("Bq3 = ", B * Q[:, 3])
println()
println("Bq4 = ", B * Q[:, 4])
println()
println("Bq5 = ", B * Q[:, 5])

#Question 5.)

#Read Data from JSON file

include("readclassjson.jl")
data = readclassjson("tomo_data.json")

y = data["y"]
npixels = data["npixels"]
N = data["N"]
line_pixels_length = data["line_pixel_lengths"]

L = reshape(line_pixels_length, npixels*npixels, N)
L = L'
pseudo_L_inverse = inv(L' * L) * L'
ls_x = pseudo_L_inverse * y
square_ls_x = reshape(ls_x, npixels, npixels)
heatmap(square_ls_x, yflip=true, aspect_ratio=:equal, color=:gist_gray,cbar=:none, framestyle=:none)