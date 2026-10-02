#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.polynomial.legendre import (
    legval,
    legder
)
from scipy.linalg import lstsq

start = time.time()

# =====================================================
# Domain = [0, 1]^3
# =====================================================
xmin, xmax = 0.0, 1.0

ymin, ymax = 0.0, 1.0

zmin, zmax = 0.0, 1.0

scale_d2x = (
    2 / (xmax - xmin)
) ** 2

scale_d2y = (
    2 / (ymax - ymin)
) ** 2

scale_d2z = (
    2 / (zmax - zmin)
) ** 2

# =====================================================
# Exact solution & PDE
# =====================================================
def u_exact(
    x,
    y,
    z
):

    return (
        np.exp(x) * (y**2)
        + (z**2 + 2.0) * np.sin(y)
    )

def rhs_A(
    x,
    y,
    z
):

    u = u_exact(x, y, z)

    return (
        (2 + y**2) * np.exp(x)
        - (z**2) * np.sin(y)
        - u**2
    )

# =====================================================
# Map physical → Legendre coordinates
# =====================================================
def map_to_leg(X):

    x = X[:, 0]

    y = X[:, 1]

    z = X[:, 2]

    xi = (
        2 * (x - xmin)
        / (xmax - xmin)
        - 1
    )

    yi = (
        2 * (y - ymin)
        / (ymax - ymin)
        - 1
    )

    zi = (
        2 * (z - zmin)
        / (zmax - zmin)
        - 1
    )

    return xi, yi, zi

# =====================================================
# 3D Legendre basis & Laplacian matrices
# =====================================================
def legendre_matrices_3d(
    xi,
    yi,
    zi,
    N
):

    Px, D2x = [], []

    Py, D2y = [], []

    Pz, D2z = [], []

    for i in range(N):

        ci = [0] * i + [1]

        d2ci = legder(
            legder(ci)
        )

        Px.append(
            legval(xi, ci)
        )

        D2x.append(
            legval(xi, d2ci)
            * scale_d2x
        )

        Py.append(
            legval(yi, ci)
        )

        D2y.append(
            legval(yi, d2ci)
            * scale_d2y
        )

        Pz.append(
            legval(zi, ci)
        )

        D2z.append(
            legval(zi, d2ci)
            * scale_d2z
        )

    cols_P = []

    cols_L = []

    for i in range(N):

        for j in range(N):

            for k in range(N):

                Pijk = (
                    Px[i]
                    * Py[j]
                    * Pz[k]
                )

                Lijk = (
                    D2x[i] * Py[j] * Pz[k]
                    + Px[i] * D2y[j] * Pz[k]
                    + Px[i] * Py[j] * D2z[k]
                )

                cols_P.append(Pijk)

                cols_L.append(Lijk)

    P = np.vstack(cols_P).T

    L = np.vstack(cols_L).T

    return P, L

# =====================================================
# Solver: returns RMS error
# =====================================================
def solve_rms_error(
    interior_pts,
    boundary_pts,
    N_basis,
    maxit=20,
    tol=1e-12,
    damping=1.0
):
    # -------------------------------------------------
    # Map to Legendre coordinates
    # -------------------------------------------------
    xi_int, yi_int, zi_int = map_to_leg(
        interior_pts
    )

    xi_b, yi_b, zi_b = map_to_leg(
        boundary_pts
    )

    # -------------------------------------------------
    # Basis matrices
    # -------------------------------------------------
    P_int, L_int = legendre_matrices_3d(
        xi_int,
        yi_int,
        zi_int,
        N_basis
    )

    P_b, _ = legendre_matrices_3d(
        xi_b,
        yi_b,
        zi_b,
        N_basis
    )

    # -------------------------------------------------
    # RHS and boundary conditions
    # -------------------------------------------------
    x = interior_pts[:, 0]

    y = interior_pts[:, 1]

    z = interior_pts[:, 2]

    A_int = rhs_A(x, y, z)

    xb = boundary_pts[:, 0]

    yb = boundary_pts[:, 1]

    zb = boundary_pts[:, 2]

    U_b = u_exact(xb, yb, zb)

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    A = np.vstack([
        L_int,
        P_b
    ])

    rhs = np.concatenate([
        A_int,
        U_b
    ])

    beta = lstsq(
        A,
        rhs,
        lapack_driver='gelsy'
    )[0]

    for it in range(maxit):

        u = P_int @ beta

        R_int = (
            L_int @ beta
            - u**2
            - A_int
        )

        R_b = (
            P_b @ beta
            - U_b
        )

        R = np.concatenate([
            R_int,
            R_b
        ])

        J_int = (
            L_int
            - 2 * (u[:, None] * P_int)
        )

        J = np.vstack([
            J_int,
            P_b
        ])

        beta_old = beta.copy()

        delta = lstsq(
            J,
            -R,
            lapack_driver='gelsy'
        )[0]

        beta += damping * delta

        rel_change = (
            np.linalg.norm(beta - beta_old)
            / np.linalg.norm(beta_old)
        )

        if rel_change < tol:
            break

    # -------------------------------------------------
    # RMS error
    # -------------------------------------------------
    U_pred = np.concatenate([
        P_int @ beta,
        P_b @ beta
    ])

    U_ex = np.concatenate([
        u_exact(x, y, z),
        u_exact(xb, yb, zb)
    ])

    return np.sqrt(
        np.mean(
            (U_pred - U_ex)**2
        )
    )

# =====================================================
# RMS error vs collocation points
# =====================================================
basis_list = [7, 8, 9, 10, 11]

datasets = [1, 2, 3, 4, 5]

colors = plt.cm.tab10.colors

plt.figure(figsize=(5.6, 4.4))

for i, N_basis in enumerate(basis_list):

    collocation_counts = []

    rms_errors = []

    for k in datasets:

        # -------------------------------------------------
        # Load datasets
        # -------------------------------------------------
        interior_pts = np.load(
            f"data/shell_i{k}.npy"
        )

        boundary_pts = np.load(
            f"data/shell_b{k}.npy"
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

        rms_errors.append(err)

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.semilogy(
        collocation_counts,
        rms_errors,
        marker="o",
        markersize=5,
        markeredgecolor="black",
        linewidth=1.3,
        color=colors[i % len(colors)],
        label=f"$N_b = {N_basis**3}$"
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
    title="Total basis functions",
    fontsize=8,
    title_fontsize=9,
    frameon=False
)

plt.grid(True)

plt.tight_layout()

# =====================================================
# Save figure
# =====================================================
plt.savefig(
    "Figure8_c.pdf",
    bbox_inches="tight"
)

plt.show()

end = time.time()

print(
    "Time taken =",
    np.round(end - start, 2),
    "s"
)


# In[ ]:




