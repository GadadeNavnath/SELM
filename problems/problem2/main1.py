#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time

# ======================================================
# Initial Parameters
# ======================================================
N = 23  # number of hidden neurons per variable
M = 50         # collocation points for Gauss-Newton iteration
M_init = 50    # collocation points for initialization
a, b = 0.0, 3.0
tol = 1e-12
maxit = 20
damping = 1.0

# Random seed for reproducibility
seed = 12

# Exact solutions
def u1_exact(x):
    return np.sin(x)

def u2_exact(x):
    return 1 + x**2

# ======================================================
# PIELM hidden-layer weights and biases
# ======================================================
np.random.seed(seed)

# Fixed random input-to-hidden weights and biases
w = np.random.uniform(-1.0, 1.0, N)
bias = np.random.uniform(-1.0, 1.0, N)

# ======================================================
# Activation function and derivatives
# ======================================================
def activation(z):
    return np.tanh(z)

def activation_d1(z):
    return 1.0 - np.tanh(z)**2

def activation_d2(z):
    return -2.0 * np.tanh(z) * (1.0 - np.tanh(z)**2)

# ======================================================
# Initialization step (linearized system)
# ======================================================
k = np.arange(M_init + 2)
x_cgl = np.cos(np.pi * k / (M_init + 1))

x_mapped = a + (b - a) * (x_cgl + 1) / 2
x_init = x_mapped[1:-1]

# Hidden-layer argument
z_init = (
    x_init[:, None] * w[None, :]
    + bias[None, :]
)

# Hidden-layer output and derivatives
Phi_init = activation(z_init)

Phi1_init = (
    activation_d1(z_init)
    * w[None, :]
)

Phi2_init = (
    activation_d2(z_init)
    * w[None, :]**2
)

S = 1 + x_init**2 + np.sin(x_init)**2

# ======================================================
# Linearized initialization system
# ======================================================
A_init = np.zeros((2*M_init + 2, 2*N))
rhs_init = np.zeros(2*M_init + 2)

for i, x in enumerate(x_init):

    # First equation:
    # u1' - u2 = cos(x) - S
    A_init[2*i, :N] = Phi1_init[i, :]
    A_init[2*i, N:] = -Phi_init[i, :]
    rhs_init[2*i] = np.cos(x) - S[i]

    # Second equation:
    # u2' = 2x - (1+x^2)sin(x)
    A_init[2*i+1, N:] = Phi1_init[i, :]
    rhs_init[2*i+1] = (
        2*x - (1+x**2)*np.sin(x)
    )

# ======================================================
# Initial conditions at x = a
# ======================================================
z_bc = a*w + bias

bc_vals = activation(z_bc)

bc_d1val = (
    activation_d1(z_bc) * w
)

A_init[2*M_init, :N] = bc_vals
rhs_init[2*M_init] = 0.0

A_init[2*M_init+1, N:] = bc_vals
rhs_init[2*M_init+1] = 1.0

# ======================================================
# Solve for initial beta
# ======================================================
t0 = time.time()

beta_init = lstsq(
    A_init,
    rhs_init,
    lapack_driver='gelsy'
)[0]

t_initial = time.time() - t0

print(f"Initialization time = {t_initial:.5f} s")
print("")

# ======================================================
# Precompute for Gauss-Newton iteration
# ======================================================
k = np.arange(M + 2)

x_cgl = np.cos(np.pi * k / (M + 1))

x_col = (
    a + (b - a) * (x_cgl[1:-1] + 1) / 2
)

# Hidden-layer argument
z_col = (
    x_col[:, None] * w[None, :]
    + bias[None, :]
)

# Hidden-layer output and derivatives
Phi_c = activation(z_col)

Phi1_c = (
    activation_d1(z_col)
    * w[None, :]
)

Phi2_c = (
    activation_d2(z_col)
    * w[None, :]**2
)

# ======================================================
# Residual and Jacobian
# ======================================================
def residual_and_jacobian(beta):

    a_coef = beta[:N]
    b_coef = beta[N:]

    # u1 and derivatives
    u1 = Phi_c @ a_coef
    u1p = Phi1_c @ a_coef

    # u2 and derivatives
    u2 = Phi_c @ b_coef
    u2p = Phi1_c @ b_coef

    F = np.zeros(2*M + 2)
    J = np.zeros((2*M + 2, 2*N))

    for i, x in enumerate(x_col):

        Sx = 1 + x**2 + np.sin(x)**2

        # --------------------------------------------------
        # First nonlinear equation:
        #
        # u1' - (cos(x) + u1^2 + u2 - S) = 0
        # --------------------------------------------------
        F[2*i] = (
            u1p[i]
            - (
                np.cos(x)
                + u1[i]**2
                + u2[i]
                - Sx
            )
        )

        J[2*i, :N] = (
            Phi1_c[i, :]
            - 2*u1[i]*Phi_c[i, :]
        )

        J[2*i, N:] = -Phi_c[i, :]

        # --------------------------------------------------
        # Second nonlinear equation:
        #
        # u2' - [2x -(1+x^2)sin(x) + u1*u2] = 0
        # --------------------------------------------------
        F[2*i+1] = (
            u2p[i]
            - (
                2*x
                - (1+x**2)*np.sin(x)
                + u1[i]*u2[i]
            )
        )

        J[2*i+1, :N] = (
            -u2[i]*Phi_c[i, :]
        )

        J[2*i+1, N:] = (
            Phi1_c[i, :]
            - u1[i]*Phi_c[i, :]
        )

    # ==================================================
    # Initial conditions
    # ==================================================

    # u1(a) = 0
    J[2*M, :N] = bc_vals
    F[2*M] = a_coef @ bc_vals

    # u2(a) = 1
    J[2*M+1, N:] = bc_vals
    F[2*M+1] = b_coef @ bc_vals - 1.0

    return F, J

# ======================================================
# Gauss-Newton iteration
# ======================================================
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

# ======================================================
# Postprocessing
# ======================================================
x_test = np.linspace(a, b, 200)

z_test = (
    x_test[:, None] * w[None, :]
    + bias[None, :]
)

Phi_test = activation(z_test)

u1_approx = Phi_test @ beta[:N]
u2_approx = Phi_test @ beta[N:]

u1_ex = u1_exact(x_test)
u2_ex = u2_exact(x_test)

abs_err_u1 = np.abs(u1_ex - u1_approx)
abs_err_u2 = np.abs(u2_ex - u2_approx)

rms_err_u1 = np.sqrt(np.mean(abs_err_u1**2))
rms_err_u2 = np.sqrt(np.mean(abs_err_u2**2))

print("")
print(
    f"Linf error (u1) = {np.max(abs_err_u1):.2e}, "
    f"RMS error = {rms_err_u1:.2e}"
)

print(
    f"Linf error (u2) = {np.max(abs_err_u2):.2e}, "
    f"RMS error = {rms_err_u2:.2e}"
)

# ======================================================
# Absolute error plot
# ======================================================
plt.figure(figsize=(6, 5))

plt.plot(
    x_test,
    abs_err_u1,
    color="black",
    linewidth=1.5,
    label=r"$|u_1 - \hat{u}_1|$"
)

plt.plot(
    x_test,
    abs_err_u2,
    color="blue",
    linewidth=1.5,
    label=r"$|u_2 - \hat{u}_2|$"
)

plt.xlabel("x", fontsize=12)
plt.ylabel("Absolute Error", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()
# plt.savefig("Figure2_a_PIELM.pdf", bbox_inches="tight")
plt.show()

# ======================================================
# Residual vs Gauss-Newton Iteration
# ======================================================
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
# plt.savefig("Figure2_r_PIELM.pdf", bbox_inches="tight")
plt.show()


# In[ ]:




