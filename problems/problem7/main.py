#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder
from scipy.linalg import lstsq
import time

# =====================================================
# Parameters
# =====================================================
maxit = 20
tol = 1e-12
damping = 1.0

# Bratu parameter
lam = 6.8081244226

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

# =====================================================
# Mapping [0,1] -> [-1,1]
# =====================================================

def to_legendre_domain(z):

    return 2*z - 1


# =====================================================
# Legendre basis
# =====================================================

N_basis = 19
NB2 = N_basis**2


def legendre_matrix_2d(x, y, N):

    xi = to_legendre_domain(x)
    eta = to_legendre_domain(y)

    Mcols = []

    for i in range(N):

        Px = legval(
            xi,
            [0]*i + [1]
        )

        for j in range(N):

            Py = legval(
                eta,
                [0]*j + [1]
            )

            Mcols.append(
                Px * Py
            )

    return np.vstack(Mcols).T


def legendre_derivative_matrices_2d(x, y, N):

    xi = to_legendre_domain(x)
    eta = to_legendre_domain(y)

    dx_cols = []
    dy_cols = []

    dxx_cols = []
    dyy_cols = []

    for i in range(N):

        coeff_i = [0]*i + [1]

        d1_i = legder(coeff_i)
        d2_i = legder(legder(coeff_i))

        Px = legval(
            xi,
            coeff_i
        )

        dPx = legval(
            xi,
            d1_i
        )

        d2Px = legval(
            xi,
            d2_i
        )

        for j in range(N):

            coeff_j = [0]*j + [1]

            d1_j = legder(coeff_j)
            d2_j = legder(legder(coeff_j))

            Py = legval(
                eta,
                coeff_j
            )

            dPy = legval(
                eta,
                d1_j
            )

            d2Py = legval(
                eta,
                d2_j
            )

            dx_cols.append(
                (2*dPx) * Py
            )

            dy_cols.append(
                Px * (2*dPy)
            )

            dxx_cols.append(
                (4*d2Px) * Py
            )

            dyy_cols.append(
                Px * (4*d2Py)
            )

    return (
        np.vstack(dx_cols).T,
        np.vstack(dy_cols).T,
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )


# =====================================================
# Precompute basis matrices
# =====================================================

P_int = legendre_matrix_2d(
    X_int,
    Y_int,
    N_basis
)

P_b = legendre_matrix_2d(
    X_b,
    Y_b,
    N_basis
)

(
    dxP_int,
    dyP_int,
    dxxP_int,
    dyyP_int
) = legendre_derivative_matrices_2d(
    X_int,
    Y_int,
    N_basis
)

(
    dxP_b,
    dyP_b,
    _,
    _
) = legendre_derivative_matrices_2d(
    X_b,
    Y_b,
    N_basis
)

# =====================================================
# Bratu problem
#
#     u_xx + u_yy + lambda exp(u) = 0
#
# Boundary condition:
#
#     u = 0 on entire boundary
# =====================================================

# =====================================================
# Linear initialization
#
# exp(u) ≈ 1 + u
#
# Therefore:
#
#     u_xx + u_yy + lambda u = -lambda
# =====================================================

t0 = time.time()

A_lin = np.vstack([
    dxxP_int
    + dyyP_int
    + lam * P_int,

    P_b
])

rhs_lin = np.concatenate([
    -lam * np.ones_like(X_int),
    np.zeros_like(X_b)
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
    f"Initialization time = {t_init:.5f} s"
)

print("")

# =====================================================
# Gauss-Newton iteration
# =====================================================

res_history = []

t_start = time.time()

for it in range(maxit):

    # -------------------------------------------------
    # Current solution and derivatives
    # -------------------------------------------------

    u_int = P_int @ beta

    uxx_int = dxxP_int @ beta

    uyy_int = dyyP_int @ beta

    # -------------------------------------------------
    # Nonlinear Bratu residual
    # -------------------------------------------------

    R_int = (
        uxx_int
        + uyy_int
        + lam * np.exp(u_int)
    )

    # -------------------------------------------------
    # Dirichlet boundary residual
    # -------------------------------------------------

    u_b = P_b @ beta

    R_b = u_b

    # -------------------------------------------------
    # Total residual
    # -------------------------------------------------

    R = np.concatenate([
        R_int,
        R_b
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
    # Jacobian
    # -------------------------------------------------

    J_int = (
        dxxP_int
        + dyyP_int
        + (
            lam
            * np.exp(u_int)
        )[:, None] * P_int
    )

    J_b = P_b

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
    # Stopping criterion
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
    f"Newton time = "
    f"{t_newton:.4f} s"
)

print(
    f"Total time = "
    f"{t_init + t_newton:.4f} s"
)

print("")


# =====================================================
# Postprocessing: comparison with P2 FEM solution
# =====================================================

fem_data = np.loadtxt(
    "fem.csv",
    delimiter=",",
    skiprows=1
)

X_fem = fem_data[:, 0]
Y_fem = fem_data[:, 1]
U_fem = fem_data[:, 2]

print("========================================")
print("FEM comparison data")
print("========================================")
print(
    f"FEM points = {len(U_fem)}"
)
print("")


# =====================================================
# Legendre-MELM prediction at FEM coordinates
# =====================================================

P_fem = legendre_matrix_2d(
    X_fem,
    Y_fem,
    N_basis
)

U_melm = P_fem @ beta


# =====================================================
# Error with respect to FEM solution
# =====================================================

abs_err = np.abs(
    U_melm - U_fem
)

rms_err = np.sqrt(
    np.mean(
        abs_err**2
    )
)

linf_err = np.max(
    abs_err
)

print("========================================")
print("Legendre-MELM vs P2 FEM")
print("========================================")

print(
    f"RMS error  = {rms_err:.3e}"
)

print(
    f"Linf error = {linf_err:.3e}"
)

print("")


# =====================================================
# FEM and MELM solution ranges
# =====================================================

print("FEM solution:")

print(
    f"Minimum = {np.min(U_fem):.6e}"
)

print(
    f"Maximum = {np.max(U_fem):.6e}"
)

print("")

print("Legendre-MELM solution:")

print(
    f"Minimum = {np.min(U_melm):.6e}"
)

print(
    f"Maximum = {np.max(U_melm):.6e}"
)

print("")


# =====================================================
# Boundary error
# =====================================================

boundary_fem = (
    np.isclose(X_fem, 0.0)
    | np.isclose(X_fem, 1.0)
    | np.isclose(Y_fem, 0.0)
    | np.isclose(Y_fem, 1.0)
)

boundary_err = abs_err[boundary_fem]

boundary_rms = np.sqrt(
    np.mean(
        boundary_err**2
    )
)

boundary_linf = np.max(
    boundary_err
)

print("========================================")
print("Boundary error")
print("========================================")

print(
    f"Boundary nodes = "
    f"{np.sum(boundary_fem)}"
)

print(
    f"Boundary RMS error  = "
    f"{boundary_rms:.3e}"
)

print(
    f"Boundary Linf error = "
    f"{boundary_linf:.3e}"
)

print("")

# =====================================================
# P2 FEM reference solution
# =====================================================

fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(
    X_fem,
    Y_fem,
    U_fem,
    color='blue',
    linewidth=0,
    alpha=0.9
)

ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$y$", fontsize=12)
ax.set_zlabel("$u(x,y)$", fontsize=12)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.xaxis.labelpad = 10
ax.yaxis.labelpad = 10
ax.zaxis.labelpad = 10

ax.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure7_a.pdf",
    bbox_inches="tight"
)

plt.show()

# =====================================================
# Absolute error contour plot
# =====================================================

plt.figure(figsize=(6, 5))

cp = plt.tricontourf(
    X_fem,
    Y_fem,
    abs_err,
    levels=40,
    cmap='viridis'
)

plt.colorbar(
    cp,
    label=r'$|u_{FEM}-\hat{u}|$'
)

plt.xlabel("$x$")

plt.ylabel("$y$")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.gca().set_aspect(
    "equal",
    adjustable="box"
)

plt.tight_layout()

plt.savefig(
    "Figure7_b.pdf",
    bbox_inches="tight"
)

plt.show()


# =====================================================
# Residual vs Newton iteration
# =====================================================

iters = np.arange(
    1,
    len(res_history) + 1
)

plt.figure(figsize=(5, 4))

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
    "Newton iteration",
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

