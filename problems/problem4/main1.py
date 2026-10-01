#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
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
    "data/chhattisgarh_interior_points.npy"
)

boundary_pts = np.load(
    "data/chhattisgarh_boundary_points.npy"
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
# PIELM parameters
# =====================================================

N_basis = 23**2

np.random.seed(12)

# Fixed random hidden-layer weights and biases
w_x = np.random.uniform(
    -1, 1, N_basis
)

w_y = np.random.uniform(
    -1, 1, N_basis
)

bias = np.random.uniform(
    -1, 1, N_basis
)


# =====================================================
# PIELM activation and derivative matrices
# =====================================================

def pielm_matrices(x, y):

    z = (
        x[:, None] * w_x[None, :]
        + y[:, None] * w_y[None, :]
        + bias[None, :]
    )

    H = np.tanh(z)

    sech2 = 1 - H**2

    Hx = (
        w_x[None, :]
        * sech2
    )

    Hy = (
        w_y[None, :]
        * sech2
    )

    Hxx = (
        -2
        * w_x[None, :]**2
        * H
        * sech2
    )

    Hyy = (
        -2
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


# =====================================================
# Assemble PIELM matrices
# =====================================================

P_int, dxP_int, dyP_int, dxx_int, dyy_int = (
    pielm_matrices(
        xi_int,
        yi_int
    )
)

P_b, dxP_b, dyP_b, dxx_b, dyy_b = (
    pielm_matrices(
        xi_b,
        yi_b
    )
)


# =====================================================
# RHS and boundary values
# =====================================================

RHS_int = f_rhs(
    xi_int,
    yi_int
)

U_b_exact = u_exact(
    xi_b,
    yi_b
)


# =====================================================
# Solve linear system
# =====================================================

t0 = time.time()

A_int = (
    dxx_int
    + dyy_int
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

t_solve = time.time() - t0


# =====================================================
# Residual norm
# =====================================================

residual_vec = (
    A_lin @ beta
    - R_lin
)

res_norm = np.linalg.norm(
    residual_vec
)

print("")
print(
    "PIELM hidden neurons =",
    N_basis
)

print(
    "Random seed = 12"
)

print(
    f"Residual norm = "
    f"{res_norm:.3e}"
)

print(
    f"Total time = "
    f"{t_solve:.4f} s"
)

print("")


# =====================================================
# Training error
# =====================================================

U_pred_train = np.vstack([
    P_int,
    P_b
]) @ beta

U_exact_train = np.concatenate([
    u_exact(
        xi_int,
        yi_int
    ),
    u_exact(
        xi_b,
        yi_b
    )
])

abs_err_train = np.abs(
    U_pred_train
    - U_exact_train
)

rms_train = np.sqrt(
    np.mean(
        abs_err_train**2
    )
)

linf_train = np.max(
    abs_err_train
)

print(
    f"Training RMS error = "
    f"{rms_train:.2e}"
)

print(
    f"Training Linf error = "
    f"{linf_train:.2e}"
)


# =====================================================
# Load test points
# =====================================================

interior_test = np.load(
    "data/chhattisgarh_interior_points_test.npy"
)

boundary_test = np.load(
    "data/chhattisgarh_boundary_points_test.npy"
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
# Test PIELM matrices
# =====================================================

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


# =====================================================
# Test predictions
# =====================================================

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
    - U_test_exact
)


# =====================================================
# Test error norms
# =====================================================

rms_test = np.sqrt(
    np.mean(
        abs_err_test**2
    )
)

linf_test = np.max(
    abs_err_test
)

print("")

print(
    f"Test RMS error = "
    f"{rms_test:.2e}"
)

print(
    f"Test Linf error = "
    f"{linf_test:.2e}"
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

plt.figure(
    figsize=(6, 5)
)

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

# plt.savefig(
#     "Figure4_b_PIELM.pdf",
#     bbox_inches="tight"
# )

plt.show()

