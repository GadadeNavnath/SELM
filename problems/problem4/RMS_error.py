#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.polynomial.legendre import legval, legder
import matplotlib.pyplot as plt

# =====================================================
# Exact solution and RHS
# =====================================================
def u_exact(x, y):

    return np.exp(-x) * (x + y**3)

def f_rhs(x, y):

    return np.exp(-x) * (x - 2 + y**3 + 6*y)

# =====================================================
# Shifted Legendre basis for [0,10] → [-1,1]
# =====================================================
def leg_shifted(n, x):

    x_hat = (x / 5.0) - 1.0

    return legval(x_hat, [0] * n + [1])

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

    x_hat = (x / 5.0) - 1.0
    y_hat = (y / 5.0) - 1.0

    scale = (1 / 5.0)**2

    for i in range(N):

        ci = [0] * i + [1]

        d2ci = legder(legder(ci))

        Px = legval(x_hat, ci)

        d2Px = scale * legval(x_hat, d2ci)

        for j in range(N):

            cj = [0] * j + [1]

            d2cj = legder(legder(cj))

            Py = legval(y_hat, cj)

            d2Py = scale * legval(y_hat, d2cj)

            dxx_cols.append(d2Px * Py)

            dyy_cols.append(Px * d2Py)

    return np.vstack(dxx_cols).T, np.vstack(dyy_cols).T

# =====================================================
# Solver returning RMS error
# =====================================================
def solve_rms_error(interior_pts, boundary_pts, N_basis):

    xi_int = interior_pts[:, 0]
    yi_int = interior_pts[:, 1]

    xi_b = boundary_pts[:, 0]
    yi_b = boundary_pts[:, 1]

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

    dxx_int, dyy_int = legendre_derivative_matrices_2d(
        xi_int,
        yi_int,
        N_basis
    )

    # -------------------------------------------------
    # RHS and boundary data
    # -------------------------------------------------
    RHS_int = f_rhs(xi_int, yi_int)

    U_b_exact = u_exact(xi_b, yi_b)

    # -------------------------------------------------
    # Linear system
    # -------------------------------------------------
    A = np.vstack([
        dxx_int + dyy_int,
        P_b
    ])

    rhs = np.concatenate([
        RHS_int,
        U_b_exact
    ])

    beta = np.linalg.lstsq(
        A,
        rhs,
        rcond=None
    )[0]

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    U_pred = np.concatenate([
        P_int @ beta,
        P_b @ beta
    ])

    U_exact_all = np.concatenate([
        u_exact(xi_int, yi_int),
        u_exact(xi_b, yi_b)
    ])

    # -------------------------------------------------
    # RMS error
    # -------------------------------------------------
    rms_error = np.sqrt(
        np.mean((U_pred - U_exact_all)**2)
    )

    return rms_error

# =====================================================
# RMS error vs collocation points
# =====================================================
basis_list = [17, 18, 19, 20, 21]

datasets = [1, 2, 3, 4, 5]

colors = plt.cm.tab10.colors

plt.figure(figsize=(5.2, 4.2))

for i, N_basis in enumerate(basis_list):

    collocation_counts = []

    errors = []

    for k in datasets:

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------
        interior_pts = np.load(
            f"data/int_points_{k}.npy"
        )

        boundary_pts = np.load(
            f"data/bnd_points_{k}.npy"
        )

        # -------------------------------------------------
        # Total collocation points
        # -------------------------------------------------
        Nc = (
            interior_pts.shape[0]
            + boundary_pts.shape[0]
        )

        # -------------------------------------------------
        # RMS error
        # -------------------------------------------------
        err = solve_rms_error(
            interior_pts,
            boundary_pts,
            N_basis
        )

        collocation_counts.append(Nc)

        errors.append(err)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.semilogy(
        collocation_counts,
        errors,
        marker="o",
        markersize=4.5,
        markeredgewidth=1.3,
        markeredgecolor="black",
        linewidth=1.2,
        color=colors[i % len(colors)],
        label=fr"$N = {N_basis}$"
    )

# =====================================================
# Figure formatting
# =====================================================
plt.xlabel("Total number of collocation points")

plt.ylabel("RMS error")

plt.legend(
    title="Basis functions per direction",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="best"
)

plt.grid(True)

plt.tight_layout()

# =====================================================
# Save figure
# =====================================================
plt.savefig(
    "Figure4_c.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




