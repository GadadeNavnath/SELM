#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time

#========================================================
# Initial Parameters
#========================================================
N = 26         # number of hidden neurons
M = 35         # collocation points for Gauss-Newton iteration
M_init = 35    # collocation points for initialization
a = np.e
b = 2*np.e      # interval [e, 2e]
tol = 1e-12
maxit = 20
damping = 1.0

# Random seed for reproducibility
seed = 12

#========================================================
# Exact solution
#========================================================
def u_exact(x):
    return np.log(np.log(x))

#========================================================
# PIELM hidden-layer weights and biases
#========================================================
np.random.seed(seed)

# Random input-to-hidden weights and biases
# These remain fixed throughout the calculation
w = np.random.uniform(-1.0, 1.0, N)
bias = np.random.uniform(-1.0, 1.0, N)

#========================================================
# Activation function and derivatives
#========================================================
def activation(z):
    return np.tanh(z)

def activation_d1(z):
    return 1.0 - np.tanh(z)**2

def activation_d2(z):
    return -2.0 * np.tanh(z) * (1.0 - np.tanh(z)**2)

#========================================================
# Step 1 (Initialization--linearized system)
#========================================================
x_init = np.linspace(a, b, M_init+2)[1:-1]

# Hidden-layer argument
z_init = x_init[:, None] * w[None, :] + bias[None, :]

# Hidden-layer output
H_init = activation(z_init)

# First derivative with respect to x
H1_init = (
    activation_d1(z_init)
    * w[None, :]
)

# Second derivative with respect to x
H2_init = (
    activation_d2(z_init)
    * w[None, :]**2
)

# Linearized equation:
# x²u'' + 1/log(x) = 0
A_init = (x_init[:, None]**2) * H2_init
rhs_init = -1/np.log(x_init)

#========================================================
# Boundary conditions at x=a
#========================================================
z_bc = a * w + bias

bc_vals = activation(z_bc)

bc_d1val = (
    activation_d1(z_bc) * w
)

A_init = np.vstack([
    A_init,
    bc_vals,
    bc_d1val
])

rhs_init = np.concatenate([
    rhs_init,
    [0.0, 1/np.e]
])

#========================================================
# Solve for initial beta
#========================================================
t0 = time.time()

beta_init = lstsq(
    A_init,
    rhs_init,
    lapack_driver='gelsy'
)[0]

t_initial = time.time() - t0

print(f"Initialization time = {t_initial:.5f} s")
print("")

#========================================================
# Step 2 (Gauss-Newton Iteration)
#========================================================
k = np.arange(M+2)

x_cgl = np.cos(np.pi * k / (M+1))

x_col = a + (b - a) * (x_cgl[1:-1] + 1) / 2

#========================================================
# Hidden-layer functions at collocation points
#========================================================
z_col = (
    x_col[:, None] * w[None, :]
    + bias[None, :]
)

Phi_c = activation(z_col)

Phi1_c = (
    activation_d1(z_col)
    * w[None, :]
)

Phi2_c = (
    activation_d2(z_col)
    * w[None, :]**2
)

#========================================================
# Residual and Jacobian
#========================================================
def residual_and_jacobian(beta):

    uprime = beta @ Phi1_c.T
    upp = beta @ Phi2_c.T

    F = np.zeros(M+2)
    J = np.zeros((M+2, N))

    # PDE residual
    F[:M] = (
        (x_col**2) * upp
        + (x_col * uprime)**2
        + 1/np.log(x_col)
    )

    # Jacobian
    J[:M, :] = (
        (x_col**2)[:, None] * Phi2_c
        + (2 * (x_col * uprime) * x_col)[:, None]
        * Phi1_c
    )

    # Boundary condition: u(a) = 0
    F[M] = beta @ bc_vals
    J[M, :] = bc_vals

    # Boundary condition: u'(a) = 1/e
    F[M+1] = beta @ bc_d1val - 1/np.e
    J[M+1, :] = bc_d1val

    return F, J

#========================================================
# Gauss-Newton iteration
#========================================================
beta = beta_init.copy()
res_history = []

t_start = time.time()

for k in range(maxit):

    F, J = residual_and_jacobian(beta)

    Fnrm = np.linalg.norm(F)
    res_history.append(Fnrm)

    print(
        f"Gauss-Newton iter {k+1:2d}: "
        f"residual = {Fnrm:.2e}"
    )

    # QR-based least-squares solution
    delta = lstsq(
        J,
        -F,
        lapack_driver='gelsy'
    )[0]

    # Store old beta
    beta_old = beta.copy()

    # Damped Gauss-Newton update
    beta += damping * delta

    # Relative change in beta
    rel_change = (
        np.linalg.norm(beta - beta_old)
        / np.linalg.norm(beta_old)
    )

    # Stopping criterion
    if rel_change < tol:
        break

t_newton = time.time() - t_start

print(f"\nGauss-Newton time = {t_newton:.4f} s")
print(f"Total time = {t_initial + t_newton:.4f} s")

#========================================================
# Postprocessing
#========================================================
x_test = np.linspace(a, b, 200)

z_test = (
    x_test[:, None] * w[None, :]
    + bias[None, :]
)

Phi_test = activation(z_test)

u_approx = Phi_test @ beta

u_ex = u_exact(x_test)

abs_err = np.abs(u_ex - u_approx)

RMS_err = np.sqrt(np.mean(abs_err**2))

print("")
print(f"Linf error = {np.max(abs_err):.2e}, RMS error = {RMS_err:.2e}")

#========================================================
# Absolute error plot
#========================================================
plt.figure(figsize=(6, 5))

plt.plot(
    x_test,
    abs_err,
    "k-",
    linewidth=1.5
)

plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.grid(True)

plt.tight_layout()

# plt.savefig(
#     "Figure1_a_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()

#========================================================
# Residual vs Gauss-Newton Iteration
#========================================================
iters = np.arange(1, len(res_history) + 1)

plt.figure(figsize=(5, 4))

plt.semilogy(
    iters,
    res_history,
    linestyle='-',
    marker='o',
    markersize=4,
    markeredgewidth=1.2,
    markeredgecolor='black',
    linewidth=1,
    color='black'
)

plt.xlabel(
    "Gauss-Newton iteration",
    fontsize=11
)

plt.ylabel(
    r"$\|R\|_2$",
    fontsize=11
)

plt.xticks(iters)
plt.grid()

plt.tight_layout()

# plt.savefig(
#     "Figure1_r_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()


# In[ ]:




