using LinearAlgebra
using Plots

#Read Data from JSON file

include("readclassjson.jl")
data = readclassjson("tomo_data.json")

y = data["y"]
npixels = data["npixels"]
N = data["N"]
line_pixels_length = data["line_pixel_lengths"]


#Question 5.)

#L = reshape(line_pixels_length, N, npixels*npixels)
L = reshape(line_pixels_length, npixels*npixels, N)
L = L'
print(rank(L' * L))
pseudo_L_inverse = inv(L' * L) * L'
ls_x = pseudo_L_inverse * y
square_ls_x = reshape(ls_x, npixels, npixels)
heatmap(square_ls_x, yflip=true, aspect_ratio=:equal, color=:gist_gray,cbar=:none, framestyle=:none)