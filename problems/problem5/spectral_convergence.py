#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder

# =====================================================
# Exact solution and right-hand side
# =====================================================

def u_exact(x, y):

    return np.log(1 + x**2 + y**2)

def rhs_pde(x, y):

    r2 = x**2 + y**2

    lap = 4.0 / (1 + r2)**2

    return lap + (1 + r2)

# =====================================================
# Load training points
# =====================================================

interior_pts = np.load(
    "data/haryana_interior_points.npy"
)

boundary_pts = np.load(
    "data/haryana_boundary_points.npy"
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
# Legendre basis
# =====================================================

def legendre_matrix_2d(x, y, N):

    cols = []

    for i in range(N):

        Px = legval(
            x,
            [0]*i + [1]
        )

        for j in range(N):

            Py = legval(
                y,
                [0]*j + [1]
            )

            cols.append(Px * Py)

    return np.vstack(cols).T

def legendre_derivative_matrices_2d(x, y, N):

    dxx_cols = []

    dyy_cols = []

    for i in range(N):

        ci = [0]*i + [1]

        d2ci = legder(
            legder(ci)
        )

        Px = legval(x, ci)

        d2Px = legval(x, d2ci)

        for j in range(N):

            cj = [0]*j + [1]

            d2cj = legder(
                legder(cj)
            )

            Py = legval(y, cj)

            d2Py = legval(y, d2cj)

            dxx_cols.append(
                d2Px * Py
            )

            dyy_cols.append(
                Px * d2Py
            )

    return (
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )

# =====================================================
# Storage
# =====================================================

N_list = np.arange(1, 25)

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

    RHS_int = rhs_pde(
        xi_int,
        yi_int
    )

    U_b_exact = u_exact(
        xi_b,
        yi_b
    )

    # -------------------------------------------------
    # Linear initialization
    # -------------------------------------------------

    A_int = (
        dxx_int
        + dyy_int
        + P_int
    )

    R_int = RHS_int - 1

    A_lin = np.vstack([
        A_int,
        P_b
    ])

    R_lin = np.concatenate([
        R_int,
        U_b_exact
    ])

    beta = np.linalg.lstsq(
        A_lin,
        R_lin,
        rcond=None
    )[0]

    # -------------------------------------------------
    # Newton iteration
    # -------------------------------------------------

    maxit = 3

    tol = 1e-10

    for it in range(maxit):

        u_int = P_int @ beta

        uxx = dxx_int @ beta

        uyy = dyy_int @ beta

        R_int = (
            uxx
            + uyy
            + np.exp(u_int)
            - RHS_int
        )

        u_b = P_b @ beta

        R_b = (
            u_b - U_b_exact
        )

        R = np.concatenate([
            R_int,
            R_b
        ])

        res = np.linalg.norm(R)

        if res < tol:

            break

        exp_u = np.exp(u_int)

        J_int = (
            dxx_int
            + dyy_int
            + (
                exp_u[:, None]
                * P_int
            )
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
        max(1, len(iters)//7)
    )
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "Figure5_s.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




