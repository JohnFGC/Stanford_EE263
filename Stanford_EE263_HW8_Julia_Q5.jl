using Distributions, Plots

# Question 5

N = 10_000

x_samples = rand(Uniform(-1, 1), N)
y_samples = rand(Uniform(-1, 1), N)
plot_z = scatter(
    x_samples, y_samples,
    markersize = 2,
    aspect_ratio = :equal,
    title = "10000 samples of z",
    xlabel = "x",
    ylabel = "y",
    legend = false,
    xlim = (-1.2, 1.2),
    ylim = (-1.2, 1.2)
)
savefig("hw8_q5ci.png")

p_samples = rand(Normal(0, 1), N)
q_samples = rand(Normal(0, 1), N)
plot_w = scatter(
    p_samples, q_samples,
    markersize = 2,
    aspect_ratio = :equal,
    title = "10000 samples of w",
    xlabel = "p",
    ylabel = "q",
    legend = false
)
savefig("hw8_q5cii.png")
