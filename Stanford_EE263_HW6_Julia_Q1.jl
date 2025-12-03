using LinearAlgebra, Plots

#Question 1.)

include("readclassjson.jl")
data = readclassjson("tempdata.json")

y_train = data["y_train"]
x_train = data["x_train"]
y_test = data["y_test"]
x_test = data["x_test"]

#Part C
function compute_F(x_array, m)
    F = zeros(length(x_array), m + 1)
    F[:, 1] .= 1.0
    for j in 1:m
        for (i, x) in enumerate(x_array)
            if x < ((j-1) * 12.0 / m)
                F[i, j + 1] = 0.0
            elseif x <= (j * 12.0 / m)
                F[i, j + 1] = m*(x/12.0) - (j-1)
            else
                F[i, j + 1] = 1.0
            end
        end
    end
    return F
end
            
m_values = [3, 6, 12, 24]
errors = zeros(length(m_values))
alpha_dict = Dict{Int, Vector{Float64}}()
for (i, m) in enumerate(m_values)
    F = compute_F(x_train, m)
    alpha = F \ y_train
    g = F*alpha
    plot(x_train, y_train, label = "y_train", seriestype=:scatter)
    plot!(x_train, g, label = "g", seriestype=:scatter)
    xlabel!("Month")
    ylabel!("Temperature(F)")
    title!("Comparing y_train and g for m=$(m)")
    savefig("hw6_q1c_m=$(m).png")

    errors[i] = norm(y_train-g)^2
    alpha_dict[m] = alpha
end

#Part D
plot(m_values, errors, label = "Continuous Extension of Error")
plot!(m_values, errors, label = "Error Points", seriestype=:scatter)
xlabel!("m")
ylabel!("Error")
title!("Minimal Squared 2-Norm Error as a function of m")
savefig("hw6_q1d.png")


#Part E
test_errors = zeros(length(m_values))
for (i, m) in enumerate(m_values)
    alpha = alpha_dict[m]
    F = compute_F(x_test, m)
    g = F * alpha
    test_errors[i] = norm(y_test-g)^2
end
plot(m_values, test_errors, label = "Continuous Extension of Test Error")
plot!(m_values, test_errors, label = "Test Error Points", seriestype=:scatter)
xlabel!("m")
ylabel!("Error")
title!("Test Error as a function of m")
savefig("hw6_q1e.png")