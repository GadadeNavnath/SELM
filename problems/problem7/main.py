#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.polynomial.legendre import (
    legval,
    legder
)

# =====================================================
# Basis & Newton settings
# =====================================================
N_basis = 10

NB = N_basis ** 3

max_iter = 4

tol = 1e-12

# =====================================================
# Domain = [0, 1]^3
# =====================================================
xmin, xmax = 0.0, 1.0

ymin, ymax = 0.0, 1.0

zmin, zmax = 0.0, 1.0

# =====================================================
# Derivative scaling factors
# =====================================================
scale_dx = 2 / (xmax - xmin)

scale_d2x = scale_dx ** 2

scale_dy = 2 / (ymax - ymin)

scale_d2y = scale_dy ** 2

scale_dz = 2 / (zmax - zmin)

scale_d2z = scale_dz ** 2

# =====================================================
# Load collocation points
# =====================================================
interior_pts = np.load(
    "data/shell_interior_points.npy"
)

boundary_pts = np.load(
    "data/shell_boundary_points.npy"
)

# =====================================================
# Number of points
# =====================================================
N_int = interior_pts.shape[0]

N_bnd = boundary_pts.shape[0]

print("Interior points =", N_int)

print("Boundary points =", N_bnd)

print("Total unknowns =", N_basis**3)

# =====================================================
# Map physical -> Legendre coordinates
# =====================================================
def map_to_leg_coords(X):

    x = X[:, 0]

    y = X[:, 1]

    z = X[:, 2]

    xi = (
        2 * (x - xmin) / (xmax - xmin)
        - 1
    )

    yi = (
        2 * (y - ymin) / (ymax - ymin)
        - 1
    )

    zi = (
        2 * (z - zmin) / (zmax - zmin)
        - 1
    )

    return xi, yi, zi

xi_int, yi_int, zi_int = map_to_leg_coords(
    interior_pts
)

xi_b, yi_b, zi_b = map_to_leg_coords(
    boundary_pts
)

# =====================================================
# Build 3D Legendre matrices
# =====================================================
def leg_matrices_3d(
    xi,
    yi,
    zi,
    N
):

    cols_P = []

    cols_dxx = []

    cols_dyy = []

    cols_dzz = []

    # -------------------------------------------------
    # 1D Legendre basis
    # -------------------------------------------------
    def leg_1d(
        pts,
        N,
        scale2
    ):

        P_list = []

        D2_list = []

        for i in range(N):

            ci = [0] * i + [1]

            d2ci = legder(
                legder(ci)
            )

            P_list.append(
                legval(pts, ci)
            )

            D2_list.append(
                legval(pts, d2ci) * scale2
            )

        return P_list, D2_list

    Px, D2x = leg_1d(
        xi,
        N,
        scale_d2x
    )

    Py, D2y = leg_1d(
        yi,
        N,
        scale_d2y
    )

    Pz, D2z = leg_1d(
        zi,
        N,
        scale_d2z
    )

    for i in range(N):

        for j in range(N):

            for k in range(N):

                cols_P.append(
                    Px[i] * Py[j] * Pz[k]
                )

                cols_dxx.append(
                    D2x[i] * Py[j] * Pz[k]
                )

                cols_dyy.append(
                    Px[i] * D2y[j] * Pz[k]
                )

                cols_dzz.append(
                    Px[i] * Py[j] * D2z[k]
                )

    P = np.vstack(cols_P).T

    dxx = np.vstack(cols_dxx).T

    dyy = np.vstack(cols_dyy).T

    dzz = np.vstack(cols_dzz).T

    return P, dxx, dyy, dzz

P_int, dxx_int, dyy_int, dzz_int = leg_matrices_3d(
    xi_int,
    yi_int,
    zi_int,
    N_basis
)

P_b, _, _, _ = leg_matrices_3d(
    xi_b,
    yi_b,
    zi_b,
    N_basis
)

# =====================================================
# PDE definitions
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

def A_xyz(
    x,
    y,
    z
):

    u_ex = u_exact(x, y, z)

    return (
        (2 + y**2) * np.exp(x)
        - (z**2) * np.sin(y)
        - u_ex**2
    )

# =====================================================
# RHS and boundary
# =====================================================
x_int = interior_pts[:, 0]

y_int = interior_pts[:, 1]

z_int = interior_pts[:, 2]

A_int = A_xyz(
    x_int,
    y_int,
    z_int
)

x_b = boundary_pts[:, 0]

y_b = boundary_pts[:, 1]

z_b = boundary_pts[:, 2]

U_b_exact = u_exact(
    x_b,
    y_b,
    z_b
)

# =====================================================
# Timing: total start
# =====================================================
t_total_start = time.time()

# =====================================================
# Linear PDE initialization
# =====================================================
t_init_start = time.time()

A_lin = np.vstack([
    dxx_int + dyy_int + dzz_int,
    P_b
])

R_lin = np.concatenate([
    A_int,
    U_b_exact
])

beta_init = np.linalg.lstsq(
    A_lin,
    R_lin,
    rcond=None
)[0]

t_init_end = time.time()

t_init = t_init_end - t_init_start

beta = beta_init.copy()

# =====================================================
# Gauss Newton iteration
# =====================================================
print("\nGauss-Newton iteration")

t_newton_start = time.time()

res_hist = []

for it in range(max_iter):

    u_int = P_int @ beta

    uxx = dxx_int @ beta

    uyy = dyy_int @ beta

    uzz = dzz_int @ beta

    R_int = (
        uxx
        + uyy
        + uzz
        - u_int**2
        - A_int
    )

    R_b = (
        P_b @ beta
        - U_b_exact
    )

    R = np.concatenate([
        R_int,
        R_b
    ])

    res = np.linalg.norm(R)

    res_hist.append(res)

    print(
        f"Iter {it+1}: ||R|| = {res:.2e}"
    )

    J_int = (
        dxx_int
        + dyy_int
        + dzz_int
        - (2 * u_int)[:, None] * P_int
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

t_newton_end = time.time()

t_newton = (
    t_newton_end - t_newton_start
)

t_total = (
    time.time() - t_total_start
)

# =====================================================
# Error evaluation
# =====================================================
u_int_pred = P_int @ beta

err_int = np.abs(
    u_int_pred
    - u_exact(x_int, y_int, z_int)
)

L2_int = np.sqrt(
    np.mean(err_int**2)
)

Linf_int = np.max(err_int)

u_b_pred = P_b @ beta

err_b = np.abs(
    u_b_pred - U_b_exact
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

u_pred = np.concatenate([
    u_int_pred,
    u_b_pred
])

abs_err = np.abs(
    u_pred - u_exact_all
)

L2_all = np.sqrt(
    np.mean(abs_err**2)
)

Linf_all = np.max(abs_err)

# =====================================================
# Training error norms
# =====================================================
print("\n===== Training results =====")

print(
    f"Interior RMS error = {L2_int:.2e}"
)

print(
    f"Interior Linf error = {Linf_int:.2e}"
)

print("")

print(
    f"Boundary RMS error = {L2_b:.2e}"
)

print(
    f"Boundary Linf error = {Linf_b:.2e}"
)

print("")

print(
    f"Total RMS error = {L2_all:.2e}"
)

print(
    f"Total Linf error = {Linf_all:.2e}"
)

# =====================================================
# Timing summary
# =====================================================
print("\n------ Timing ------")

print(
    f"Initial time : "
    f"{t_init:.2f} s"
)

print(
    f"Newton iterations time : "
    f"{t_newton:.2f} s"
)

print(
    f"Total runtime : "
    f"{t_total:.4f} s"
)
print("\n===========Test results==========")
# =====================================================
# Load test datasets
# =====================================================
int_test = np.load(
    "data/shell_interior_points_test.npy"
)

bnd_test = np.load(
    "data/shell_boundary_points_test.npy"
)

X_test = np.vstack([
    int_test,
    bnd_test
])

# =====================================================
# Map to Legendre coordinates
# =====================================================
xi_t, yi_t, zi_t = map_to_leg_coords(
    X_test
)

# =====================================================
# Build basis
# =====================================================
P_test, _, _, _ = leg_matrices_3d(
    xi_t,
    yi_t,
    zi_t,
    N_basis
)

# =====================================================
# Prediction & error
# =====================================================
u_pred = P_test @ beta

x_t = X_test[:, 0]

y_t = X_test[:, 1]

z_t = X_test[:, 2]

u_ex = u_exact(
    x_t,
    y_t,
    z_t
)

err_test = np.abs(
    u_pred - u_ex
)

# =====================================================
# Error norms
# =====================================================
RMS_test = np.sqrt(
    np.mean(err_test**2)
)

Linf_test = np.max(err_test)

print("")

# =====================================================
# Test dataset information
# =====================================================
N_int_test = int_test.shape[0]

N_bnd_test = bnd_test.shape[0]

print(
    "Interior test points =",
    N_int_test
)

print(
    "Boundary test points =",
    N_bnd_test
)

print("")

# =====================================================
# Test error norms
# =====================================================
print(
    f"RMS error = {RMS_test:.2e}"
)

print(
    f"Linf error = {Linf_test:.2e}"
)
# =====================================================
# Absolute error distribution
# =====================================================
fig = plt.figure(figsize=(6, 5))

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

ax.set_box_aspect([1, 1, 1])

plt.tight_layout()

plt.savefig(
    "Figure7_b.pdf",
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Residual convergence plot
# =====================================================
iters = np.arange(
    1,
    len(res_hist) + 1
)

plt.figure(figsize=(5, 4))

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

plt.savefig(
    "Figure7_r.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




