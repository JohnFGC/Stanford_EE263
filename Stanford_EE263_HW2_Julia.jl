using LinearAlgebra

#Read Data from JSON file

include("readclassjson.jl")
data = readclassjson("color_perception_data.json")

wavelength = data["wavelength"]
test_light = data["test_light"]
l_coefficients = data["L_coefficients"]
m_coefficients = data["M_coefficients"]
s_coefficients = data["S_coefficients"]
red_coefficients = data["R_phosphor"]
green_coefficients = data["G_phosphor"]
blue_coefficients = data["B_phosphor"]
tungsten = data["tungsten"]
sunlight = data["sunlight"]


#Question 2c.)
#Use the equation derived in 2b to solve for [a1, a2, a3]
#[a1, a2, a3] = B^(-1) *Ap_test
A = vcat(l_coefficients', m_coefficients', s_coefficients')

Au = A * red_coefficients
Av = A * green_coefficients
Aw = A * blue_coefficients

B = hcat(Au, Av, Aw)
println("[a1, a2, a3] = ", inv(B)*A*test_light)


#Question 2d.)
#We want both objects to be visually similar with tungsten bulb
#and different with sunlight
#Goal: tungsten .* r_object1 + z = tungsten .* r_object2, where z is in nullspace
#r_object2(i) = r_object1(i) + [z(i) / tungsten(i)]

r_object1 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
r_object2 = similar(r_object1)
y = nullspace(A)
for i=1:20
    r_object2[i] = (y[i,1] / tungsten[i]) + r_object1[i]
end
println("r_object1 = ", r_object1)
println("r_object2 = ", r_object2)

#Now we verify by taking the norms of the objects' difference in reflection for each of the lights
#We want tungsten norm to be nearly 0 which means vector in nullspace and indistinguishable
#and the sunlight norm to be nonzero which means objects are distinguishable only in this light
println("Tungsten Difference Norm: ", norm(A * (tungsten .* (r_object1 - r_object2))))
println("Sunlight Difference Norm: ", norm(A * (sunlight .* (r_object1 - r_object2))))
println("As seen by the norms, object 1 and object 2 are indistinguishable when lit by a tungsten bulb but not when lit by sunlight")