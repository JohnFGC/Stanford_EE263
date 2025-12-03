using LinearAlgebra, Distributions, Plots

#Question 4.)

#Part a
q1 = [500.0, 0.0]
q2 = [0.0, 500.0]
q3 = [500.0, -200.0]

A = [(q1' / norm(q1)); 
     (q2' / norm(q2));
     (q3' / norm(q3))]

sigma_x = I(2)
sigma_w = diagm([0.1, 1.0, 1.0])

Lmmse = sigma_x * A' * inv(A * sigma_x * A' + sigma_w)
println("L_mmse = ", Lmmse)

#Part b
mu_x = [2.0, 3.0]
mu_w = zeros(3)

x = rand(MvNormal(mu_x, sigma_x))
w = rand(MvNormal(mu_w, sigma_w))
y = A * x + w

mu_mmse = mu_x + Lmmse * (y - A * mu_x)
sigma_mmse = sigma_x - Lmmse * A * sigma_x

theta = range(0, 2*pi, length=400)
c = sqrt(4.605)           # chi2 for 90%
eigvals, eigvecs = eigen(sigma_mmse)

ellipse = [mu_mmse .+ c * eigvecs * diagm(sqrt.(eigvals)) * [cos(t), sin(t)] for t in theta]
xs = [p[1] for p in ellipse]
ys = [p[2] for p in ellipse]

plot(xs, ys, aspect_ratio=:equal, label="90% Ellipse")
scatter!([mu_mmse[1]], [mu_mmse[2]], label="Estimated x")
scatter!([x[1]], [x[2]], label="True x", marker=:star)
savefig("hw9_q4b.png")

#Part c
N = 500
x_samples = rand(MvNormal(mu_x, sigma_x), N)
w_samples = rand(MvNormal(mu_w, sigma_w), N)
A_pinv = pinv(A)
mmse_errors = zeros(2, N)
ls_errors = zeros(2, N)

for k in 1:N
    x = x_samples[:, k]
    w = w_samples[:, k]
    y = A*x + w

    mmse_errors[:, k] = x - (mu_x + Lmmse * (y - A * mu_x))
    ls_errors[:, k] = x - A_pinv*y
end

sigma_mmse_err = cov(eachcol(mmse_errors))
eigvals, eigvecs = eigen(sigma_mmse_err)
theta = range(0, 2π, length=400)
c = sqrt(4.605)   # chi2 for 90%

ellipse_mmse = [c * eigvecs * diagm(sqrt.(eigvals)) * [cos(t), sin(t)] for t in theta]
xs_mmse = [p[1] for p in ellipse_mmse]
ys_mmse = [p[2] for p in ellipse_mmse]

plot(xs_mmse, ys_mmse, aspect_ratio=:equal, label="MMSE 90% Ellipse")
scatter!(mmse_errors[1, :], mmse_errors[2, :], alpha=0.5, label="MMSE Errors")
savefig("hw9_q4c.png")

#Part e
sigma_ls_err = cov(eachcol(ls_errors))
eigvals_ls, eigvecs_ls = eigen(sigma_ls_err)
theta = range(0, 2π, length=400)
c = sqrt(4.605)   # chi2 for 90%

ellipse_ls = [c * eigvecs_ls * diagm(sqrt.(eigvals_ls)) * [cos(t), sin(t)] for t in theta]
xs_ls = [p[1] for p in ellipse_ls]
ys_ls = [p[2] for p in ellipse_ls]

plot(xs_ls, ys_ls, aspect_ratio=:equal, label="LS 90% Ellipse")
scatter!(ls_errors[1, :], ls_errors[2, :], alpha=0.5, label="LS Errors")
savefig("hw9_q4e.png")