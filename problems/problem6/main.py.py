#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder

# ======================================================
# Load training points
# ======================================================

interior_pts = np.load(
    "data/star_interior_points.npy"
)

boundary_pts = np.load(
    "data/star_boundary_points.npy"
)

print("Interior points :", interior_pts.shape[0])
print("Boundary points :", boundary_pts.shape[0])

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# ======================================================
# Legendre tensor-product basis
# ======================================================

N_basis = 20

def legendre_matrix_2d(x, y, N):

    cols = []

    for i in range(N):

        Px = legval(x, [0]*i + [1])

        for j in range(N):

            Py = legval(y, [0]*j + [1])

            cols.append(Px * Py)

    return np.vstack(cols).T

def legendre_derivative_matrices_2d(x, y, N):

    dxx_cols = []
    dyy_cols = []

    for i in range(N):

        ci = [0]*i + [1]
        d2ci = legder(legder(ci))

        Px = legval(x, ci)
        d2Px = legval(x, d2ci)

        for j in range(N):

            cj = [0]*j + [1]
            d2cj = legder(legder(cj))

            Py = legval(y, cj)
            d2Py = legval(y, d2cj)

            dxx_cols.append(d2Px * Py)
            dyy_cols.append(Px * d2Py)

    return np.vstack(dxx_cols).T, np.vstack(dyy_cols).T

# ======================================================
# Precompute matrices
# ======================================================

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

# ======================================================
# Exact solution and RHS
# ======================================================

def u_exact(x, y):

    return (
        np.sin(np.pi*x)
        * np.sin(np.pi*y)
    )

def rhs_pde(x, y):

    return (
        2*np.pi**2
        * np.sin(np.pi*x)
        * np.sin(np.pi*y)
        +
        (
            np.sin(np.pi*x)
            * np.sin(np.pi*y)
        )**3
    )

RHS_int = rhs_pde(
    xi_int,
    yi_int
)

U_b_exact = u_exact(
    xi_b,
    yi_b
)

# ======================================================
# Linear initialization
# ======================================================

t0 = time.time()

A_int = -(dxx_int + dyy_int)

A_lin = np.vstack([
    A_int,
    P_b
])

R_lin = np.concatenate([
    RHS_int,
    U_b_exact
])

beta = np.linalg.lstsq(
    A_lin,
    R_lin,
    rcond=None
)[0]

init_time = time.time() - t0

print("\n========================================")
print(f"Initialization time = {init_time:.4f} s")
print("")
# ======================================================
# Newton iteration
# ======================================================

maxit = 4
tol = 1e-12

res_history = []

t0 = time.time()

for it in range(maxit):

    u_int = P_int @ beta

    uxx = dxx_int @ beta
    uyy = dyy_int @ beta

    # ==================================================
    # Residual
    # ==================================================

    R_int = (
        -(uxx + uyy)
        + u_int**3
        - RHS_int
    )

    R_b = (
        (P_b @ beta)
        - U_b_exact
    )

    R = np.concatenate([
        R_int,
        R_b
    ])

    res = np.linalg.norm(R)

    res_history.append(res)

    print(f"Iter {it+1:02d}: ||R|| = {res:.3e}")

    if res < tol:

        print("Newton converged")

        break

    # ==================================================
    # Jacobian
    # ==================================================

    J_int = (
        -(dxx_int + dyy_int)
        +
        (3*u_int**2)[:, None] * P_int
    )

    J = np.vstack([
        J_int,
        P_b
    ])

    delta = np.linalg.lstsq(
        J,
        -R,
        rcond=None
    )[0]

    beta += delta

newton_time = time.time() - t0

print("")
print(f"Newton time = {newton_time:.4f} s")
print(f"Total time  = {init_time + newton_time:.4f} s")
# ======================================================
# Postprocessing
# ======================================================

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


# ======================================================
# Error norms
# ======================================================

l2_err = np.sqrt(
    np.mean(abs_err**2)
)

linf_err = np.max(abs_err)

print("\n========================================")
print("Training error")
print("========================================")
print(f"RMS error  = {l2_err:.2e}")
print(f"Linf error = {linf_err:.2e}")

# ======================================================
# Load TEST points
# ======================================================

interior_test = np.load(
    "data/star_interior_points_test.npy"
)

boundary_test = np.load(
    "data/star_boundary_points_test.npy"
)

print("\n========================================")
print("Test data")
print("========================================")
print("Test interior :", interior_test.shape[0])
print("Test boundary :", boundary_test.shape[0])

xi_test_int = interior_test[:, 0]
yi_test_int = interior_test[:, 1]

xi_test_b = boundary_test[:, 0]
yi_test_b = boundary_test[:, 1]

# ======================================================
# Test basis matrices
# ======================================================

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

# ======================================================
# Predictions on TEST set
# ======================================================

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

# ======================================================
# Combine predictions and exact
# ======================================================

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

# ======================================================
# Test error norms
# ======================================================

l2_test = np.sqrt(
    np.mean(abs_err_test**2)
)

linf_test = np.max(
    abs_err_test
)

print("\n--------------------------------------")
print("Test error")
print(f"TEST RMS error  = {l2_test:.2e}")
print(f"TEST Linf error = {linf_test:.2e}")

# ======================================================
# Absolute Error Distribution
# ======================================================

X_all = np.concatenate([
    xi_test_int,
    xi_test_b
])

Y_all = np.concatenate([
    yi_test_int,
    yi_test_b
])

plt.figure(figsize=(6, 5))

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

plt.savefig(
    "Figure6_b.pdf",
    bbox_inches="tight"
)

plt.show()

# ======================================================
# Residual convergence plot
# ======================================================

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

plt.savefig(
    "Figure6_r.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




