#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import matplotlib.pyplot as plt
import time

# =====================================================
# Exact solution and right-hand side
# =====================================================

def u_exact(x, y):

    return np.exp(-x) * (x + y**3)

def f_rhs(x, y):

    return np.exp(-x) * (x - 2 + y**3 + 6*y)

# =====================================================
# Load training points
# =====================================================

interior_pts = np.load(
    "chhattisgarh_interior_points.npy"
)

boundary_pts = np.load(
    "chhattisgarh_boundary_points.npy"
)

print(
    "Training interior points:",
    interior_pts.shape[0]
)

print(
    "Training boundary points:",
    boundary_pts.shape[0]
)

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# =====================================================
# Shifted Legendre basis
# =====================================================

N_basis = 20

def leg_shifted(n, x):

    x_hat = x / 5.0 - 1.0

    coeffs = [0]*n + [1]

    return legval(x_hat, coeffs)

def legendre_matrix_2d(x, y, N):

    cols = []

    for i in range(N):

        Px = leg_shifted(i, x)

        for j in range(N):

            Py = leg_shifted(j, y)

            cols.append(Px * Py)

    return np.vstack(cols).T

def legendre_derivative_matrices_2d(x, y, N):

    dxx_cols = []

    dyy_cols = []

    x_hat = x / 5.0 - 1.0

    y_hat = y / 5.0 - 1.0

    scale = (1 / 5.0)**2

    for i in range(N):

        ci = [0]*i + [1]

        d2ci = legder(legder(ci))

        Px = legval(x_hat, ci)

        d2Px = scale * legval(x_hat, d2ci)

        for j in range(N):

            cj = [0]*j + [1]

            d2cj = legder(legder(cj))

            Py = legval(y_hat, cj)

            d2Py = scale * legval(y_hat, d2cj)

            dxx_cols.append(d2Px * Py)

            dyy_cols.append(Px * d2Py)

    return (
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )

# =====================================================
# Assemble basis matrices
# =====================================================

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

dxx_int, dyy_int = (
    legendre_derivative_matrices_2d(
        xi_int,
        yi_int,
        N_basis
    )
)

# =====================================================
# RHS and boundary values
# =====================================================

RHS_int = f_rhs(xi_int, yi_int)

U_b_exact = u_exact(xi_b, yi_b)

# =====================================================
# Solve linear system
# =====================================================

t0 = time.time()

A_int = dxx_int + dyy_int

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

t_solve = time.time() - t0

# =====================================================
# Residual norm
# =====================================================

residual_vec = A_lin @ beta - R_lin

res_norm = np.linalg.norm(residual_vec)
print("")
print(
    f"Residual norm = {res_norm:.3e}"
)

print(
    f"Total time = {t_solve:.4f} s"
)
print("")
# =====================================================
# Training error
# =====================================================

X_all_train = np.concatenate([
    xi_int,
    xi_b
])

Y_all_train = np.concatenate([
    yi_int,
    yi_b
])

U_pred_train = np.vstack([
    P_int,
    P_b
]) @ beta

U_exact_train = np.concatenate([
    u_exact(xi_int, yi_int),
    u_exact(xi_b, yi_b)
])

abs_err_train = np.abs(
    U_pred_train - U_exact_train
)

rms_train = np.sqrt(
    np.mean(abs_err_train**2)
)

linf_train = np.max(abs_err_train)

print(
    f"Training RMS error = {rms_train:.2e}"
)

print(
    f"Training Linf error = {linf_train:.2e}"
)

# =====================================================
# Load test points
# =====================================================

interior_test = np.load(
    "chhattisgarh_interior_points_test.npy"
)

boundary_test = np.load(
    "chhattisgarh_boundary_points_test.npy"
)

print("")

print(
    "Test interior points:",
    interior_test.shape[0]
)

print(
    "Test boundary points:",
    boundary_test.shape[0]
)

xi_test_int = interior_test[:, 0]
yi_test_int = interior_test[:, 1]

xi_test_b = boundary_test[:, 0]
yi_test_b = boundary_test[:, 1]

# =====================================================
# Test basis matrices
# =====================================================

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

# =====================================================
# Test predictions
# =====================================================

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
# =====================================================
# Test error norms
# =====================================================

rms_test = np.sqrt(
    np.mean(abs_err_test**2)
)

linf_test = np.max(abs_err_test)

print("")

print(
    f"Test RMS error = {rms_test:.2e}"
)

print(
    f"Test Linf error = {linf_test:.2e}"
)
# =====================================================
# Absolute error distribution
# =====================================================

X_all_test = np.concatenate([
    xi_test_int,
    xi_test_b
])

Y_all_test = np.concatenate([
    yi_test_int,
    yi_test_b
])

plt.figure(figsize=(6, 5))

sc = plt.scatter(
    X_all_test,
    Y_all_test,
    c=abs_err_test,
    cmap="viridis",
    s=1,
    zorder=1
)

plt.colorbar(
    sc,
    label=r"$|u - \hat{u}|$",
    shrink=0.95
)

boundary_closed = np.vstack([
    boundary_test,
    boundary_test[0]
])

plt.plot(
    boundary_closed[:, 0],
    boundary_closed[:, 1],
    linestyle='-',
    linewidth=0.5,
    color='black',
    alpha=0.6,
    zorder=2
)

plt.xlabel("$x$")

plt.ylabel("$y$")

plt.xlim(0, 10)

plt.ylim(0, 10)

plt.gca().set_aspect(
    "equal",
    adjustable="box"
)

plt.tight_layout()

plt.savefig(
    "Figure4_b.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




