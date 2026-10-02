#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder
from scipy.linalg import lstsq
import time

maxit = 20
tol = 1e-12
damping = 1.0
# ==========================================================
# Load interior & boundary points
# ==========================================================
interior_pts = np.load("data/haryana_interior_points.npy")

boundary_pts = np.load("data/haryana_boundary_points.npy")

print("")
print("Interior points:", interior_pts.shape[0])
print("Boundary points:", boundary_pts.shape[0])
print("")
if interior_pts is None or boundary_pts is None:
    raise RuntimeError("Interior/boundary points not found.")

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# ==========================================================
# Legendre basis and derivatives
# ==========================================================
N_basis = 21

def legendre_matrix_2d(x, y, N):

    Mcols = []

    for i in range(N):

        Px = legval(x, [0] * i + [1])

        for j in range(N):

            Py = legval(y, [0] * j + [1])

            Mcols.append(Px * Py)

    return np.vstack(Mcols).T


def legendre_derivative_matrices_2d(x, y, N):

    dxx_cols = []

    dyy_cols = []

    for i in range(N):

        ci = [0] * i + [1]

        d2ci = legder(legder(ci))

        Px = legval(x, ci)

        d2Px = legval(x, d2ci)

        for j in range(N):

            cj = [0] * j + [1]

            d2cj = legder(legder(cj))

            Py = legval(y, cj)

            d2Py = legval(y, d2cj)

            dxx_cols.append(d2Px * Py)

            dyy_cols.append(Px * d2Py)

    return np.vstack(dxx_cols).T, np.vstack(dyy_cols).T


# ==========================================================
# Precompute basis matrices
# ==========================================================
P_int = legendre_matrix_2d(
    xi_int,
    yi_int,
    N_basis
)

P_b = legendre_matrix_2d(
    xi_b,
    yi_b,
    N_basis
)

dxx_int, dyy_int = legendre_derivative_matrices_2d(
    xi_int,
    yi_int,
    N_basis
)

dxx_b, dyy_b = legendre_derivative_matrices_2d(
    xi_b,
    yi_b,
    N_basis
)

# ==========================================================
# RHS and exact solution
# ==========================================================
def u_exact(x, y):

    return np.log(1 + x**2 + y**2)


def rhs_pde(x, y):

    r2 = x**2 + y**2

    lap = 4.0 / (1 + r2)**2

    return lap + (1 + r2)


RHS_int = rhs_pde(xi_int, yi_int)

U_b_exact = u_exact(xi_b, yi_b)

# ==========================================================
# Initialization: solve linear PDE
# ==========================================================
t0 = time.time()

A_int = dxx_int + dyy_int + P_int

R_int = RHS_int - 1

A_lin = np.vstack([
    A_int,
    P_b
])

R_lin = np.concatenate([
    R_int,
    U_b_exact
], axis=0)

beta = lstsq(
    A_lin,
    R_lin,
    lapack_driver='gelsy'
)[0]

t_init = time.time() - t0

print(f"Initialization time = {t_init:.4f} s")
print("")

# ==========================================================
# Gauss-Newton iteration for nonlinear PDE
# ==========================================================
res_history = []

t_start = time.time()

for it in range(maxit):

    # ------------------------------------------------------
    # Current solution
    # ------------------------------------------------------
    u_int = P_int @ beta

    uxx = dxx_int @ beta
    uyy = dyy_int @ beta

    # ------------------------------------------------------
    # Residual
    # ------------------------------------------------------
    R_int = (
        uxx
        + uyy
        + np.exp(u_int)
        - RHS_int
    )

    u_b = P_b @ beta

    R_b = u_b - U_b_exact

    R = np.concatenate([
        R_int,
        R_b
    ])

    # ------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------
    exp_u = np.exp(u_int)

    J_int = (
        dxx_int
        + dyy_int
        + exp_u[:, None] * P_int
    )

    J = np.vstack([
        J_int,
        P_b
    ])

    # ------------------------------------------------------
    # QR-based least-squares Gauss-Newton step
    # ------------------------------------------------------
    beta_old = beta.copy()

    delta = lstsq(
        J,
        -R,
        lapack_driver='gelsy'
    )[0]

    # ------------------------------------------------------
    # Damped update
    # ------------------------------------------------------
    beta += damping * delta

    # ------------------------------------------------------
    # Relative coefficient change
    # ------------------------------------------------------
    rel_change = (
        np.linalg.norm(beta - beta_old)
        / np.linalg.norm(beta_old)
    )

    # ------------------------------------------------------
    # Residual after update
    # ------------------------------------------------------
    u_new = P_int @ beta

    R_int_new = (
        dxx_int @ beta
        + dyy_int @ beta
        + np.exp(u_new)
        - RHS_int
    )

    R_b_new = P_b @ beta - U_b_exact

    R_new = np.concatenate([
        R_int_new,
        R_b_new
    ])

    res_norm = np.linalg.norm(R_new)

    res_history.append(res_norm)

    print(
        f"Iter {it+1:02d}: "
        f"||R|| = {res_norm:.3e}, "
        f"rel. change = {rel_change:.3e}"
    )

    # ------------------------------------------------------
    # Stopping criterion: beta only
    # ------------------------------------------------------
    if rel_change < tol:

        print(
            f"Converged at iteration {it+1}: "
            f"relative coefficient change = "
            f"{rel_change:.3e}"
        )

        break

t_newton = time.time() - t_start
print("")
print(
    f"Newton iteration time = "
    f"{t_newton:.4f} s"
)

print(
    f"Total time = "
    f"{t_init + t_newton:.4f} s"
)

print("")

# ==========================================================
# Postprocessing
# ==========================================================
X_all = np.concatenate([
    xi_int,
    xi_b
])

Y_all = np.concatenate([
    yi_int,
    yi_b
])

U_pred_all = np.vstack([
    P_int,
    P_b
]) @ beta

U_exact_all = np.concatenate([
    u_exact(xi_int, yi_int),
    u_exact(xi_b, yi_b)
])

abs_err = np.abs(
    U_pred_all - U_exact_all
)

# ==========================================================
# Error norms
# ==========================================================
l2_err = np.sqrt(
    np.mean(abs_err**2)
)

linf_err = np.max(abs_err)

print(
    f"L2 error = {l2_err:.2e}, "
    f"Linf error = {linf_err:.2e}"
)

# ==========================================================
# Load TEST points
# ==========================================================
interior_test = np.load(
    "data/haryana_interior_points_test.npy"
)

boundary_test = np.load(
    "data/haryana_boundary_points_test.npy"
)

xi_test_int = interior_test[:, 0]

yi_test_int = interior_test[:, 1]

xi_test_b = boundary_test[:, 0]

yi_test_b = boundary_test[:, 1]

print("---------------------------------------------------")

print(
    "Test interior:",
    interior_test.shape[0]
)

print(
    "Test boundary:",
    boundary_test.shape[0]
)

# ==========================================================
# Test basis matrices
# ==========================================================
P_test_int = legendre_matrix_2d(
    xi_test_int,
    yi_test_int,
    N_basis
)

P_test_b = legendre_matrix_2d(
    xi_test_b,
    yi_test_b,
    N_basis
)

# ==========================================================
# Predictions on TEST set
# ==========================================================
U_test_pred_int = P_test_int @ beta

U_test_pred_b = P_test_b @ beta

U_test_exact_int = u_exact(
    xi_test_int,
    yi_test_int
)

U_test_exact_b = u_exact(
    xi_test_b,
    yi_test_b
)

# ==========================================================
# Combine predictions and exact values
# ==========================================================
U_test_pred = np.concatenate([
    U_test_pred_int,
    U_test_pred_b
])

U_test_exact = np.concatenate([
    U_test_exact_int,
    U_test_exact_b
])

abs_err_test = np.abs(
    U_test_pred - U_test_exact
)

# ==========================================================
# TEST error norms
# ==========================================================
l2_test = np.sqrt(
    np.mean(abs_err_test**2)
)

linf_test = np.max(abs_err_test)

print(
    f"TEST RMS error  = {l2_test:.2e}"
)

print(
    f"TEST Linf error = {linf_test:.2e}"
)

# ==========================================================
# Absolute Error Distribution (TEST)
# ==========================================================
X_all = np.concatenate([
    xi_test_int,
    xi_test_b
])

Y_all = np.concatenate([
    yi_test_int,
    yi_test_b
])

plt.figure(figsize=(6, 5))

# ==========================================================
# Absolute error scatter
# ==========================================================
sc = plt.scatter(
    X_all,
    Y_all,
    c=abs_err_test,
    cmap="viridis",
    s=1,
    zorder=1
)

plt.colorbar(
    sc,
    label=r"$|u - \hat{u}|$",
    shrink=0.90
)

# ==========================================================
# Boundary curve
# ==========================================================
bx = xi_test_b

by = yi_test_b

plt.plot(
    bx,
    by,
    linestyle="-",
    linewidth=0.5,
    color="black",
    alpha=0.6,
    zorder=2
)

# ==========================================================
# Figure formatting
# ==========================================================
plt.xlabel("$x$")

plt.ylabel("$y$")

plt.xlim(-1, 1)

plt.ylim(-1, 1)

ax = plt.gca()

ax.set_aspect(
    "equal",
    adjustable="box"
)

plt.tight_layout()

# ==========================================================
# Save figure
# ==========================================================
plt.savefig(
    "Figure5_b.pdf",
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# Residual convergence plot
# ==========================================================
iters = np.arange(
    1,
    len(res_history) + 1
)

plt.figure(figsize=(5, 4))

plt.semilogy(
    iters,
    res_history,
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

# ==========================================================
# Save figure
# ==========================================================
plt.savefig(
    "Figure5_r.pdf",
    bbox_inches="tight"
)

plt.show()

