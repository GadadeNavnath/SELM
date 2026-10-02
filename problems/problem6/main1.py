#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time

# ======================================================
# Parameters
# ======================================================

maxit = 20
tol = 1e-12
damping = 1.0

N_basis = 400

np.random.seed(12)


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
# PIELM random hidden-layer parameters
# ======================================================

w_x = np.random.uniform(
    -1.0,
    1.0,
    N_basis
)

w_y = np.random.uniform(
    -1.0,
    1.0,
    N_basis
)

bias = np.random.uniform(
    -1.0,
    1.0,
    N_basis
)

print(
    "PIELM hidden neurons:",
    N_basis
)


# ======================================================
# PIELM activation and derivative matrices
# ======================================================

def pielm_matrices(x, y):

    z = (
        x[:, None] * w_x[None, :]
        +
        y[:, None] * w_y[None, :]
        +
        bias[None, :]
    )

    H = np.tanh(z)

    sech2 = 1.0 - H**2

    Hx = (
        w_x[None, :]
        * sech2
    )

    Hy = (
        w_y[None, :]
        * sech2
    )

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

    return (
        H,
        Hx,
        Hy,
        Hxx,
        Hyy
    )


# ======================================================
# Precompute PIELM matrices
# ======================================================

P_int, dxP_int, dyP_int, dxxP_int, dyyP_int = (
    pielm_matrices(
        xi_int,
        yi_int
    )
)

P_b, dxP_b, dyP_b, dxxP_b, dyyP_b = (
    pielm_matrices(
        xi_b,
        yi_b
    )
)


# ======================================================
# Exact solution and RHS
# ======================================================

def u_exact(x, y):

    return (
        np.sin(np.pi*x)
        *
        np.sin(np.pi*y)
    )


def rhs_pde(x, y):

    return (
        2*np.pi**2
        *
        np.sin(np.pi*x)
        *
        np.sin(np.pi*y)
        +
        (
            np.sin(np.pi*x)
            *
            np.sin(np.pi*y)
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

A_int = -(
    dxxP_int
    +
    dyyP_int
)

A_lin = np.vstack([
    A_int,
    P_b
])

R_lin = np.concatenate([
    RHS_int,
    U_b_exact
])

beta = lstsq(
    A_lin,
    R_lin,
    lapack_driver='gelsy'
)[0]

init_time = time.time() - t0

print("")
print(
    f"Initialization time = "
    f"{init_time:.4f} s"
)
print("")


# ======================================================
# Gauss-Newton iteration
# ======================================================

res_history = []

t0 = time.time()

for it in range(maxit):

    # --------------------------------------------------
    # Current solution and derivatives
    # --------------------------------------------------

    u_int = P_int @ beta

    uxx = dxxP_int @ beta

    uyy = dyyP_int @ beta

    # --------------------------------------------------
    # Nonlinear residual
    # --------------------------------------------------

    R_int = (
        -(uxx + uyy)
        +
        u_int**3
        -
        RHS_int
    )

    # --------------------------------------------------
    # Boundary residual
    # --------------------------------------------------

    u_b = P_b @ beta

    R_b = (
        u_b
        -
        U_b_exact
    )

    # --------------------------------------------------
    # Complete residual
    # --------------------------------------------------

    R = np.concatenate([
        R_int,
        R_b
    ])

    res = np.linalg.norm(R)

    res_history.append(res)

    print(
        f"Iter {it+1:02d}: "
        f"||R|| = {res:.3e}"
    )

    # --------------------------------------------------
    # Jacobian
    #
    # R = -u_xx - u_yy + u^3 - f
    #
    # dR/dbeta =
    # -Hxx - Hyy + 3u^2 H
    # --------------------------------------------------

    J_int = (
        -(
            dxxP_int
            +
            dyyP_int
        )
        +
        (
            3.0 * u_int**2
        )[:, None]
        * P_int
    )

    J = np.vstack([
        J_int,
        P_b
    ])

    # --------------------------------------------------
    # QR-based Gauss-Newton step
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

    beta += damping * delta

    # --------------------------------------------------
    # Relative coefficient-change stopping criterion
    # --------------------------------------------------

    rel_change = (
        np.linalg.norm(
            beta - beta_old
        )
        /
        np.linalg.norm(beta_old)
    )

    if rel_change < tol:

        print(
            f"Converged at iteration "
            f"{it+1}: "
            f"relative coefficient change = "
            f"{rel_change:.3e}"
        )

        break


newton_time = time.time() - t0

print("")
print(
    f"Newton time = "
    f"{newton_time:.4f} s"
)

print(
    f"Total time  = "
    f"{init_time + newton_time:.4f} s"
)


# ======================================================
# Postprocessing: Training points
# ======================================================

U_pred_all = np.vstack([
    P_int,
    P_b
]) @ beta

U_exact_all = np.concatenate([
    u_exact(
        xi_int,
        yi_int
    ),
    u_exact(
        xi_b,
        yi_b
    )
])

abs_err = np.abs(
    U_pred_all
    -
    U_exact_all
)


# ======================================================
# Training error
# ======================================================

l2_err = np.sqrt(
    np.mean(
        abs_err**2
    )
)

linf_err = np.max(
    abs_err
)

print("")
print("========================================")
print("Training error")
print("========================================")
print(
    f"RMS error  = "
    f"{l2_err:.2e}"
)

print(
    f"Linf error = "
    f"{linf_err:.2e}"
)


# ======================================================
# Load TEST points
# ======================================================

interior_test = np.load(
    "data/star_interior_points_test.npy"
)

boundary_test = np.load(
    "data/star_boundary_points_test.npy"
)

print("")
print("========================================")
print("Test data")
print("========================================")

print(
    "Test interior :",
    interior_test.shape[0]
)

print(
    "Test boundary :",
    boundary_test.shape[0]
)

xi_test_int = interior_test[:, 0]
yi_test_int = interior_test[:, 1]

xi_test_b = boundary_test[:, 0]
yi_test_b = boundary_test[:, 1]


# ======================================================
# Test PIELM matrices
# ======================================================

P_test_int, _, _, _, _ = (
    pielm_matrices(
        xi_test_int,
        yi_test_int
    )
)

P_test_b, _, _, _, _ = (
    pielm_matrices(
        xi_test_b,
        yi_test_b
    )
)


# ======================================================
# Test predictions
# ======================================================

U_test_pred_int = (
    P_test_int @ beta
)

U_test_pred_b = (
    P_test_b @ beta
)

U_test_exact_int = u_exact(
    xi_test_int,
    yi_test_int
)

U_test_exact_b = u_exact(
    xi_test_b,
    yi_test_b
)


# ======================================================
# Combine test predictions
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
    U_test_pred
    -
    U_test_exact
)


# ======================================================
# Test error norms
# ======================================================

l2_test = np.sqrt(
    np.mean(
        abs_err_test**2
    )
)

linf_test = np.max(
    abs_err_test
)

print("")
print("--------------------------------------")
print("Test error")

print(
    f"TEST RMS error  = "
    f"{l2_test:.2e}"
)

print(
    f"TEST Linf error = "
    f"{linf_test:.2e}"
)


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

plt.figure(
    figsize=(6, 5)
)

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

# plt.savefig(
#     "Figure6_b_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()


# ======================================================
# Residual convergence plot
# ======================================================

iters = np.arange(
    1,
    len(res_history) + 1
)

plt.figure(
    figsize=(5, 4)
)

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

# plt.savefig(
#     "Figure6_r_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()

