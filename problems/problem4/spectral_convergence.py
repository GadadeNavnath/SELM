#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import matplotlib.pyplot as plt

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
    "data/chhattisgarh_interior_points.npy"
)

boundary_pts = np.load(
    "data/chhattisgarh_boundary_points.npy"
)

print(
    "Interior points:",
    interior_pts.shape[0]
)

print(
    "Boundary points:",
    boundary_pts.shape[0]
)

xi_int = interior_pts[:, 0]
yi_int = interior_pts[:, 1]

xi_b = boundary_pts[:, 0]
yi_b = boundary_pts[:, 1]

# =====================================================
# Shifted Legendre basis
# =====================================================

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
# Storage
# =====================================================

N_list = np.arange(1, 21)

rms_errors = []

# =====================================================
# Loop over basis functions
# =====================================================

for N_basis in N_list:

    # -------------------------------------------------
    # Basis matrices
    # -------------------------------------------------

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

    # -------------------------------------------------
    # RHS and boundary values
    # -------------------------------------------------

    RHS_int = f_rhs(
        xi_int,
        yi_int
    )

    U_b_exact = u_exact(
        xi_b,
        yi_b
    )

    # -------------------------------------------------
    # Linear system
    # -------------------------------------------------

    A_int = dxx_int + dyy_int

    A_lin = np.vstack([
        A_int,
        P_b
    ])

    R_lin = np.concatenate([
        RHS_int,
        U_b_exact
    ])

    # -------------------------------------------------
    # Solve system
    # -------------------------------------------------

    beta = np.linalg.lstsq(
        A_lin,
        R_lin,
        rcond=None
    )[0]

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------

    U_pred_all = np.vstack([
        P_int,
        P_b
    ]) @ beta

    U_exact_all = np.concatenate([
        u_exact(xi_int, yi_int),
        u_exact(xi_b, yi_b)
    ])

    # -------------------------------------------------
    # RMS error
    # -------------------------------------------------

    rms = np.sqrt(
        np.mean(
            (U_pred_all - U_exact_all)**2
        )
    )

    rms_errors.append(rms)

# =====================================================
# Spectral convergence plot
# =====================================================

iters = N_list

rms_errors_safe = np.maximum(
    rms_errors,
    1e-16
)

# =====================================================
# Exponential fit
# =====================================================

coeffs = np.polyfit(
    iters,
    np.log(rms_errors_safe),
    1
)

fit_curve = (
    np.exp(coeffs[1])
    * np.exp(coeffs[0] * iters)
)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(5, 4))

plt.semilogy(
    iters,
    rms_errors,
    linestyle='-',
    marker='o',
    markersize=4,
    markeredgewidth=1.2,
    markeredgecolor='black',
    linewidth=1,
    color='black',
    label='RMS error'
)

plt.semilogy(
    iters,
    fit_curve,
    linestyle='-',
    linewidth=1,
    color='blue',
    label='Exponential fit'
)

plt.xlabel(
    "Number of basis functions per direction",
    fontsize=11
)

plt.ylabel(
    "RMS error",
    fontsize=11
)

plt.xticks(
    np.arange(
        1,
        len(iters) + 1,
        max(1, len(iters)//6)
    )
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "Figure4_s.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




