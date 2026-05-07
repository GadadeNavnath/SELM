#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder

# =====================================================
# Exact solution and RHS
# =====================================================
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

# =====================================================
# Legendre basis
# =====================================================
def legendre_matrix_2d(x, y, N):

    cols = []

    for i in range(N):

        Px = legval(
            x,
            [0] * i + [1]
        )

        for j in range(N):

            Py = legval(
                y,
                [0] * j + [1]
            )

            cols.append(Px * Py)

    return np.vstack(cols).T

def legendre_derivative_matrices_2d(x, y, N):

    dxx_cols = []

    dyy_cols = []

    for i in range(N):

        ci = [0] * i + [1]

        d2ci = legder(
            legder(ci)
        )

        Px = legval(x, ci)

        d2Px = legval(x, d2ci)

        for j in range(N):

            cj = [0] * j + [1]

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
# Solver returning RMS error
# =====================================================
def solve_rms_error(
    interior_pts,
    boundary_pts,
    N_basis,
    maxit=5,
    tol=1e-12
):

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
    A_int = -(
        dxx_int
        + dyy_int
    )

    A = np.vstack([
        A_int,
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
    # Newton iteration
    # -------------------------------------------------
    for _ in range(maxit):

        u_int = P_int @ beta

        uxx = dxx_int @ beta

        uyy = dyy_int @ beta

        R_int = (
            -(uxx + uyy)
            + u_int**3
            - RHS_int
        )

        R_b = (
            P_b @ beta
            - U_b_exact
        )

        R = np.concatenate([
            R_int,
            R_b
        ])

        if np.linalg.norm(R) < tol:

            break

        J_int = (
            -(dxx_int + dyy_int)
            + (
                3 * u_int**2
            )[:, None] * P_int
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
        np.mean(
            (U_pred - U_exact_all)**2
        )
    )

    return rms_error

# =====================================================
# RMS error vs collocation points
# =====================================================
basis_list = [16, 18, 20, 22, 24]

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
plt.xlabel(
    "Total number of collocation points"
)

plt.ylabel(
    "RMS error"
)

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
    "Figure6_c.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




