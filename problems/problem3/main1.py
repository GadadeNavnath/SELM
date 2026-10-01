#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
import time


# =====================================================
# Parameters
# =====================================================
maxit = 20
tol = 1e-12
damping = 1.0

# =====================================================
# PIELM parameters
# =====================================================
seed = 12

np.random.seed(seed)

N_basis = 19
N_hidden = N_basis**2

# =====================================================
# Chebyshev–Gauss–Lobatto points mapped to [0,1]
# =====================================================

Nx, Ny = 20, 20

k = np.arange(Nx)
x_cgl = np.cos(np.pi * k / (Nx - 1))

k = np.arange(Ny)
y_cgl = np.cos(np.pi * k / (Ny - 1))

x = 0.5 * (x_cgl + 1)
y = 0.5 * (y_cgl + 1)

X, Y = np.meshgrid(x, y)

Xf = X.flatten()
Yf = Y.flatten()

# =====================================================
# Interior and boundary points
# =====================================================

boundary = (
    (Xf == 0)
    | (Xf == 1)
    | (Yf == 0)
    | (Yf == 1)
)

interior = ~boundary

X_int = Xf[interior]
Y_int = Yf[interior]

X_b = Xf[boundary]
Y_b = Yf[boundary]

print("")
print("Interior points =", X_int.shape[0])
print("Boundary points =", X_b.shape[0])
print("")
print("PIELM hidden neurons =", N_hidden)
print("")


# =====================================================
# Random PIELM hidden-layer parameters
# =====================================================

w_x = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)

w_y = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
)

bias = np.random.uniform(
    -1.0,
    1.0,
    N_hidden
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

    H1 = 1.0 - H**2

    H2 = -2.0 * H * H1

    # First derivatives
    Hx = H1 * w_x[None, :]
    Hy = H1 * w_y[None, :]

    # Second derivatives
    Hxx = H2 * (w_x[None, :]**2)
    Hyy = H2 * (w_y[None, :]**2)

    return (
        H,
        Hx,
        Hy,
        Hxx,
        Hyy
    )


# =====================================================
# Precompute PIELM matrices
# =====================================================

(
    H_int,
    Hx_int,
    Hy_int,
    Hxx_int,
    Hyy_int
) = pielm_matrices(
    X_int,
    Y_int
)

(
    H_b,
    Hx_b,
    Hy_b,
    Hxx_b,
    Hyy_b
) = pielm_matrices(
    X_b,
    Y_b
)


# =====================================================
# Exact solution and RHS
# =====================================================

def rhs_func(x, y):

    return (
        np.sin(np.pi*x)
        * (
            2
            - (np.pi**2)*(y**2)
            + 2*(y**3)*np.sin(np.pi*x)
        )
    )


def exact_u(x, y):

    return (y**2) * np.sin(np.pi*x)


def exact_dudy_top(x):

    return 2*np.sin(np.pi*x)


RHS_int = rhs_func(
    X_int,
    Y_int
)

U_b_exact = exact_u(
    X_b,
    Y_b
)


# =====================================================
# Boundary classification
# =====================================================

mask_top = np.isclose(
    Y_b,
    1.0
)

mask_other = ~mask_top


# =====================================================
# Initialization
#
# Linearization:
#
#     u_xx + u_yy + u*u_y = RHS
#
# About u = 0:
#
#     u_xx + u_yy = RHS
# =====================================================

t0 = time.time()

A_lin = np.vstack([
    Hxx_int + Hyy_int,
    H_b[mask_other],
    Hy_b[mask_top]
])

rhs_lin = np.concatenate([
    RHS_int,
    U_b_exact[mask_other],
    exact_dudy_top(
        X_b[mask_top]
    )
])

# =====================================================
# QR-based least-squares solution
# =====================================================

beta = lstsq(
    A_lin,
    rhs_lin,
    lapack_driver='gelsy'
)[0]

t_init = time.time() - t0

print(
    f"Initialization time = "
    f"{t_init:.5f} s"
)

print("")


# =====================================================
# Gauss--Newton iteration
# =====================================================

res_history = []

t_start = time.time()

for it in range(maxit):

    # -------------------------------------------------
    # Current solution and derivatives
    # -------------------------------------------------

    u_int = H_int @ beta

    uy_int = Hy_int @ beta

    uxx_int = Hxx_int @ beta

    uyy_int = Hyy_int @ beta

    # -------------------------------------------------
    # Nonlinear PDE residual
    # -------------------------------------------------

    R_int = (
        uxx_int
        + uyy_int
        + u_int * uy_int
        - RHS_int
    )

    # -------------------------------------------------
    # Boundary residual
    # -------------------------------------------------

    u_b = H_b @ beta

    dudy_b = Hy_b @ beta

    R_b_dir = (
        u_b[mask_other]
        - U_b_exact[mask_other]
    )

    R_b_neu = (
        dudy_b[mask_top]
        - exact_dudy_top(
            X_b[mask_top]
        )
    )

    # -------------------------------------------------
    # Total residual
    # -------------------------------------------------

    R = np.concatenate([
        R_int,
        R_b_dir,
        R_b_neu
    ])

    res_norm = np.linalg.norm(R)

    res_history.append(
        res_norm
    )

    print(
        f"Iter {it+1:02d}: "
        f"||R|| = {res_norm:.3e}"
    )

    # -------------------------------------------------
    # Jacobian of nonlinear PDE
    #
    # d/d beta [
    #     u_xx + u_yy + u*u_y
    # ]
    #
    # = Hxx + Hyy
    #   + u_y H
    #   + u Hy
    # -------------------------------------------------

    J_nl_int = (
        uy_int[:, None] * H_int
        + u_int[:, None] * Hy_int
    )

    J_int = (
        Hxx_int
        + Hyy_int
        + J_nl_int
    )

    # -------------------------------------------------
    # Boundary Jacobian
    # -------------------------------------------------

    J_b = np.vstack([
        H_b[mask_other],
        Hy_b[mask_top]
    ])

    # -------------------------------------------------
    # Complete Jacobian
    # -------------------------------------------------

    J = np.vstack([
        J_int,
        J_b
    ])

    # -------------------------------------------------
    # Newton correction
    # -------------------------------------------------

    beta_old = beta.copy()

    delta = lstsq(
        J,
        -R,
        lapack_driver='gelsy'
    )[0]

    # -------------------------------------------------
    # Damped update
    # -------------------------------------------------

    beta += damping * delta

    # -------------------------------------------------
    # Relative coefficient-change stopping criterion
    # -------------------------------------------------

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


t_newton = time.time() - t_start

print("")
print(
    f"Gauss-Newton time = "
    f"{t_newton:.4f} s"
)

print(
    f"Total time = "
    f"{t_init + t_newton:.4f} s"
)

print("")


# =====================================================
# Postprocessing
# =====================================================

N_testx, N_testy = 100, 100

x_test = np.linspace(
    0,
    1,
    N_testx
)

y_test = np.linspace(
    0,
    1,
    N_testy
)

X_test, Y_test = np.meshgrid(
    x_test,
    y_test
)

Xf_test = X_test.flatten()
Yf_test = Y_test.flatten()


# =====================================================
# PIELM prediction at test points
# =====================================================

(
    H_test,
    Hx_test,
    Hy_test,
    Hxx_test,
    Hyy_test
) = pielm_matrices(
    Xf_test,
    Yf_test
)

U_pred_flat = H_test @ beta

U_pred = U_pred_flat.reshape(
    N_testy,
    N_testx
)

U_exact = exact_u(
    X_test,
    Y_test
)

abs_err = np.abs(
    U_pred - U_exact
)


# =====================================================
# Error norms
# =====================================================

rms_err = np.sqrt(
    np.mean(
        abs_err**2
    )
)

linf_err = np.max(
    abs_err
)

print(
    f"Linf error = {linf_err:.2e}, "
    f"RMS error = {rms_err:.2e}"
)


# =====================================================
# 3D solution comparison
# =====================================================

fig = plt.figure(
    figsize=(6, 5)
)

ax = fig.add_subplot(
    111,
    projection='3d'
)

exact_color = 'blue'
pred_color = 'black'

ax.plot_wireframe(
    X_test,
    Y_test,
    U_exact,
    color=exact_color,
    linewidth=1
)

ax.plot_wireframe(
    X_test,
    Y_test,
    U_pred,
    color=pred_color,
    linewidth=1,
    linestyle='--'
)

ax.set_xlabel(
    "$x$",
    fontsize=12
)

ax.set_ylabel(
    "$y$",
    fontsize=12
)

ax.set_zlabel(
    "$u(x,y)$",
    fontsize=12
)

ax.legend(
    handles=[
        plt.Line2D(
            [],
            [],
            color=exact_color,
            linewidth=1,
            label='Exact'
        ),
        plt.Line2D(
            [],
            [],
            color=pred_color,
            linewidth=1,
            linestyle='--',
            label='Approximate'
        )
    ],
    loc='upper left',
    fontsize=11,
    frameon=False
)

ax.grid(True)

ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10

plt.tight_layout()

# plt.savefig(
#     "Figure3_a1.pdf",
#     bbox_inches="tight"
# )

plt.show()


# =====================================================
# Absolute error contour plot
# =====================================================

plt.figure(
    figsize=(6, 5)
)

cp = plt.contourf(
    X_test,
    Y_test,
    abs_err,
    levels=40,
    cmap='viridis'
)

plt.colorbar(
    cp,
    label=r'$|u-\hat{u}|$'
)

plt.xlabel("$x$")
plt.ylabel("$y$")

plt.tight_layout()

# plt.savefig(
#     "Figure3_b1.pdf",
#     bbox_inches="tight"
# )

plt.show()


# =====================================================
# Residual vs Gauss--Newton Iteration
# =====================================================

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
    linestyle='-',
    marker='o',
    markersize=4,
    markeredgewidth=1.2,
    markeredgecolor='black',
    linewidth=1,
    color='black'
)

plt.xlabel(
    "Gauss--Newton iteration",
    fontsize=11
)

plt.ylabel(
    r"$\|R\|_2$",
    fontsize=11
)

plt.xticks(
    iters
)

plt.grid()

plt.tight_layout()

# plt.savefig(
#     "Figure3_r1.pdf",
#     bbox_inches="tight"
# )

plt.show()

