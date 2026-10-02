#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq


# ======================================================
# PIELM & Newton settings
# ======================================================

N_hidden = 1331

seed = 12

max_iter = 20

tol = 1e-12

damping = 1.0

np.random.seed(seed)


# ======================================================
# Domain = [0, 1]^3
# ======================================================

xmin, xmax = 0.0, 1.0

ymin, ymax = 0.0, 1.0

zmin, zmax = 0.0, 1.0


# ======================================================
# Load collocation points
# ======================================================

interior_pts = np.load(
    "data/shell_interior_points.npy"
)

boundary_pts = np.load(
    "data/shell_boundary_points.npy"
)

N_int = interior_pts.shape[0]

N_bnd = boundary_pts.shape[0]

print("Interior points =", N_int)

print("Boundary points =", N_bnd)

print("PIELM hidden neurons =", N_hidden)

print("Random seed =", seed)


# ======================================================
# Coordinates
# ======================================================

x_int = interior_pts[:, 0]

y_int = interior_pts[:, 1]

z_int = interior_pts[:, 2]

x_b = boundary_pts[:, 0]

y_b = boundary_pts[:, 1]

z_b = boundary_pts[:, 2]


# ======================================================
# Fixed random PIELM hidden-layer parameters
# ======================================================

w_x = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)

w_y = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)

w_z = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)

bias = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)


# ======================================================
# PIELM basis and second derivatives
# ======================================================

def pielm_matrices_3d(
    x,
    y,
    z
):

    # --------------------------------------------------
    # Hidden-layer argument
    # --------------------------------------------------

    Z = (
        x[:, None] * w_x[None, :]
        + y[:, None] * w_y[None, :]
        + z[:, None] * w_z[None, :]
        + bias[None, :]
    )

    # --------------------------------------------------
    # tanh activation
    # --------------------------------------------------

    H = np.tanh(Z)

    sech2 = 1.0 - H**2

    # --------------------------------------------------
    # Second derivatives
    #
    # H_xx = -2 w_x^2 H (1-H^2)
    # H_yy = -2 w_y^2 H (1-H^2)
    # H_zz = -2 w_z^2 H (1-H^2)
    # --------------------------------------------------

    Hxx = (
        -2.0
        * w_x[None, :]**2
        * H
        * sech2
    )

    Hyy = (
        -2.0
        * w_y[None, :]**2
        * H
        * sech2
    )

    Hzz = (
        -2.0
        * w_z[None, :]**2
        * H
        * sech2
    )

    return H, Hxx, Hyy, Hzz


# ======================================================
# Precompute PIELM matrices
# ======================================================

P_int, dxx_int, dyy_int, dzz_int = (
    pielm_matrices_3d(
        x_int,
        y_int,
        z_int
    )
)

P_b, _, _, _ = (
    pielm_matrices_3d(
        x_b,
        y_b,
        z_b
    )
)


# ======================================================
# Exact solution and RHS
# ======================================================

def u_exact(
    x,
    y,
    z
):

    return (
        np.exp(x) * (y**2)
        + (z**2 + 2.0) * np.sin(y)
    )


def A_xyz(
    x,
    y,
    z
):

    u_ex = u_exact(
        x,
        y,
        z
    )

    return (
        (2.0 + y**2) * np.exp(x)
        - (z**2) * np.sin(y)
        - u_ex**2
    )


# ======================================================
# RHS and boundary values
# ======================================================

A_int = A_xyz(
    x_int,
    y_int,
    z_int
)

U_b_exact = u_exact(
    x_b,
    y_b,
    z_b
)


# ======================================================
# Timing: total solver
# ======================================================

t_total_start = time.time()


# ======================================================
# Linear PDE initialization
# ======================================================

t_init_start = time.time()

A_lin = np.vstack([
    dxx_int
    + dyy_int
    + dzz_int,

    P_b
])

R_lin = np.concatenate([
    A_int,
    U_b_exact
])

beta_init = lstsq(
    A_lin,
    R_lin,
    lapack_driver='gelsy'
)[0]

t_init_end = time.time()

t_init = (
    t_init_end
    - t_init_start
)

beta = beta_init.copy()


print("")
print("========================================")
print("PIELM initialization")
print("========================================")
print(
    f"Initialization time = "
    f"{t_init:.4f} s"
)


# ======================================================
# Gauss-Newton iteration
# ======================================================

print("")
print("Gauss-Newton iteration")
print("")

t_newton_start = time.time()

res_hist = []

for it in range(max_iter):

    # --------------------------------------------------
    # Current solution and derivatives
    # --------------------------------------------------

    u_int = P_int @ beta

    uxx = dxx_int @ beta

    uyy = dyy_int @ beta

    uzz = dzz_int @ beta

    # --------------------------------------------------
    # Nonlinear PDE residual
    #
    # R = u_xx + u_yy + u_zz - u^2 - A
    # --------------------------------------------------

    R_int = (
        uxx
        + uyy
        + uzz
        - u_int**2
        - A_int
    )

    # --------------------------------------------------
    # Boundary residual
    # --------------------------------------------------

    R_b = (
        P_b @ beta
        - U_b_exact
    )

    # --------------------------------------------------
    # Complete residual
    # --------------------------------------------------

    R = np.concatenate([
        R_int,
        R_b
    ])

    res = np.linalg.norm(R)

    res_hist.append(res)

    # --------------------------------------------------
    # Jacobian
    #
    # dR/dbeta =
    # Hxx + Hyy + Hzz - 2u H
    # --------------------------------------------------

    J_int = (
        dxx_int
        + dyy_int
        + dzz_int
        - (2.0 * u_int)[:, None]
        * P_int
    )

    J = np.vstack([
        J_int,
        P_b
    ])

    # --------------------------------------------------
    # Gauss-Newton / QR least-squares step
    # --------------------------------------------------

    beta_old = beta.copy()

    delta = lstsq(
        J,
        -R,
        lapack_driver='gelsy'
    )[0]

    # --------------------------------------------------
    # Damped update
    # --------------------------------------------------

    beta += (
        damping
        * delta
    )

    # --------------------------------------------------
    # Relative coefficient-change criterion
    # --------------------------------------------------

    rel_change = (
        np.linalg.norm(
            beta - beta_old
        )
        /
        np.linalg.norm(beta_old)
    )

    print(
        f"Iter {it+1:02d}: "
        f"||R|| = {res:.3e}, "
        f"rel. change = {rel_change:.3e}"
    )

    if rel_change < tol:

        print(
            f"Converged at iteration {it+1}: "
            f"relative coefficient change = "
            f"{rel_change:.3e}"
        )

        break


t_newton_end = time.time()

t_newton = (
    t_newton_end
    - t_newton_start
)

t_total = (
    time.time()
    - t_total_start
)


# ======================================================
# Training error
# ======================================================

u_int_pred = P_int @ beta

err_int = np.abs(
    u_int_pred
    - u_exact(
        x_int,
        y_int,
        z_int
    )
)

L2_int = np.sqrt(
    np.mean(err_int**2)
)

Linf_int = np.max(err_int)


u_b_pred = P_b @ beta

err_b = np.abs(
    u_b_pred
    - U_b_exact
)

L2_b = np.sqrt(
    np.mean(err_b**2)
)

Linf_b = np.max(err_b)


x_all = np.concatenate([
    x_int,
    x_b
])

y_all = np.concatenate([
    y_int,
    y_b
])

z_all = np.concatenate([
    z_int,
    z_b
])

u_exact_all = u_exact(
    x_all,
    y_all,
    z_all
)

u_pred_all = np.concatenate([
    u_int_pred,
    u_b_pred
])

abs_err = np.abs(
    u_pred_all
    - u_exact_all
)

L2_all = np.sqrt(
    np.mean(abs_err**2)
)

Linf_all = np.max(abs_err)


# ======================================================
# Training results
# ======================================================

print("")
print("========================================")
print("Training results")
print("========================================")

print(
    f"Interior RMS error = "
    f"{L2_int:.2e}"
)

print(
    f"Interior Linf error = "
    f"{Linf_int:.2e}"
)

print("")

print(
    f"Boundary RMS error = "
    f"{L2_b:.2e}"
)

print(
    f"Boundary Linf error = "
    f"{Linf_b:.2e}"
)

print("")

print(
    f"Total RMS error = "
    f"{L2_all:.2e}"
)

print(
    f"Total Linf error = "
    f"{Linf_all:.2e}"
)


# ======================================================
# Timing summary
# ======================================================

print("")
print("========================================")
print("Timing")
print("========================================")

print(
    f"Initialization time : "
    f"{t_init:.4f} s"
)

print(
    f"Newton iteration time : "
    f"{t_newton:.4f} s"
)

print(
    f"Total solver time : "
    f"{t_total:.4f} s"
)


# ======================================================
# Load TEST datasets
# ======================================================

print("")
print("========================================")
print("Test results")
print("========================================")

int_test = np.load(
    "data/shell_interior_points_test.npy"
)

bnd_test = np.load(
    "data/shell_boundary_points_test.npy"
)

print(
    "Interior test points =",
    int_test.shape[0]
)

print(
    "Boundary test points =",
    bnd_test.shape[0]
)


# ======================================================
# Combine test points
# ======================================================

X_test = np.vstack([
    int_test,
    bnd_test
])

x_t = X_test[:, 0]

y_t = X_test[:, 1]

z_t = X_test[:, 2]


# ======================================================
# Test PIELM basis
# ======================================================

P_test, _, _, _ = (
    pielm_matrices_3d(
        x_t,
        y_t,
        z_t
    )
)


# ======================================================
# Test prediction
# ======================================================

u_pred_test = P_test @ beta

u_exact_test = u_exact(
    x_t,
    y_t,
    z_t
)

err_test = np.abs(
    u_pred_test
    - u_exact_test
)


# ======================================================
# Test error norms
# ======================================================

RMS_test = np.sqrt(
    np.mean(err_test**2)
)

Linf_test = np.max(
    err_test
)

print("")

print(
    f"TEST RMS error = "
    f"{RMS_test:.2e}"
)

print(
    f"TEST Linf error = "
    f"{Linf_test:.2e}"
)


# ======================================================
# Absolute error distribution
# ======================================================

fig = plt.figure(
    figsize=(6, 5)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

sc = ax.scatter(
    x_t,
    y_t,
    z_t,
    c=err_test,
    cmap="viridis",
    s=1
)

cbar = plt.colorbar(
    sc,
    fraction=0.046,
    pad=0.04,
    shrink=0.8
)

cbar.set_label(
    r"$|u - \hat{u}|$",
    fontsize=11
)

ax.set_xlabel(
    r"$x$",
    fontsize=11
)

ax.set_ylabel(
    r"$y$",
    fontsize=11
)

ax.set_zlabel(
    r"$z$",
    fontsize=11
)

ax.set_box_aspect(
    [1, 1, 1]
)

plt.tight_layout()

# plt.savefig(
#     "Figure8_b_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()


# ======================================================
# Residual convergence plot
# ======================================================

iters = np.arange(
    1,
    len(res_hist) + 1
)

plt.figure(
    figsize=(5, 4)
)

plt.semilogy(
    iters,
    res_hist,
    linestyle="-",
    marker="o",
    markersize=4,
    markeredgewidth=1.2,
    markeredgecolor="black",
    linewidth=1,
    color="black"
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
#     "Figure8_r_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()

