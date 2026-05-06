import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder
import time

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

boundary = (Xf == 0) | (Xf == 1) | (Yf == 0) | (Yf == 1)

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
# Legendre basis and derivatives
# =====================================================

N_basis = 19
NB2 = N_basis**2

def legendre_matrix_2d(x, y, N):

    xi = to_legendre_domain(x)
    eta = to_legendre_domain(y)

    Mcols = []

    for i in range(N):

        Px = legval(xi, [0]*i + [1])

        for j in range(N):

            Py = legval(eta, [0]*j + [1])

            Mcols.append(Px * Py)

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

        Px = legval(xi, coeff_i)
        dPx = legval(xi, d1_i)
        d2Px = legval(xi, d2_i)

        for j in range(N):

            coeff_j = [0]*j + [1]

            d1_j = legder(coeff_j)
            d2_j = legder(legder(coeff_j))

            Py = legval(eta, coeff_j)
            dPy = legval(eta, d1_j)
            d2Py = legval(eta, d2_j)

            dx_cols.append((2*dPx) * Py)
            dy_cols.append(Px * (2*dPy))

            dxx_cols.append((4*d2Px) * Py)
            dyy_cols.append(Px * (4*d2Py))

    return (
        np.vstack(dx_cols).T,
        np.vstack(dy_cols).T,
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )

# =====================================================
# Precompute basis matrices
# =====================================================

P_int = legendre_matrix_2d(X_int, Y_int, N_basis)

P_b = legendre_matrix_2d(X_b, Y_b, N_basis)

dxP_int, dyP_int, dxxP_int, dyyP_int = (
    legendre_derivative_matrices_2d(X_int, Y_int, N_basis)
)

dxP_b, dyP_b, _, _ = (
    legendre_derivative_matrices_2d(X_b, Y_b, N_basis)
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

RHS_int = rhs_func(X_int, Y_int)

U_b_exact = exact_u(X_b, Y_b)

# =====================================================
# Boundary classification
# =====================================================

mask_top = np.isclose(Y_b, 1.0)

mask_other = ~mask_top

# =====================================================
# Initialization
# =====================================================

t0 = time.time()

A_lin = np.vstack([
    dxxP_int + dyyP_int,
    P_b[mask_other],
    dyP_b[mask_top]
])

rhs_lin = np.concatenate([
    RHS_int,
    U_b_exact[mask_other],
    exact_dudy_top(X_b[mask_top])
])

beta = np.linalg.lstsq(A_lin, rhs_lin, rcond=None)[0]

t_init = time.time() - t0

print(f"Initialization time = {t_init:.5f} s")
print("")

# =====================================================
# Gauss--Newton iteration
# =====================================================

maxit = 5

tol = 1e-10

res_history = []

t_start = time.time()

for it in range(maxit):

    u_int = P_int @ beta

    uy_int = dyP_int @ beta

    uxx_int = dxxP_int @ beta

    uyy_int = dyyP_int @ beta

    R_int = (
        uxx_int
        + uyy_int
        + u_int * uy_int
        - RHS_int
    )

    u_b = P_b @ beta

    dudy_b = dyP_b @ beta

    R_b_dir = u_b[mask_other] - U_b_exact[mask_other]

    R_b_neu = (
        dudy_b[mask_top]
        - exact_dudy_top(X_b[mask_top])
    )

    R = np.concatenate([R_int, R_b_dir, R_b_neu])

    res_norm = np.linalg.norm(R)

    res_history.append(res_norm)

    print(f"Iter {it+1:02d}: ||R|| = {res_norm:.3e}")

    if res_norm < tol:

        print(f"Converged at iteration {it+1}")

        break

    J_nl_int = (
        np.diag(uy_int) @ P_int
        + np.diag(u_int) @ dyP_int
    )

    J_int = dxxP_int + dyyP_int + J_nl_int

    J_b = np.vstack([
        P_b[mask_other],
        dyP_b[mask_top]
    ])

    J = np.vstack([J_int, J_b])

    delta = np.linalg.lstsq(J, -R, rcond=None)[0]

    beta += delta

t_newton = time.time() - t_start

print("")
print(f"Gauss-Newton time = {t_newton:.4f} s")
print(f"Total time = {t_init + t_newton:.4f} s")
print("")

# =====================================================
# Postprocessing
# =====================================================

N_testx, N_testy = 100, 100

x_test = np.linspace(0, 1, N_testx)

y_test = np.linspace(0, 1, N_testy)

X_test, Y_test = np.meshgrid(x_test, y_test)

Xf_test = X_test.flatten()

Yf_test = Y_test.flatten()

U_pred_flat = (
    legendre_matrix_2d(Xf_test, Yf_test, N_basis)
    @ beta
)

U_pred = U_pred_flat.reshape(N_testx, N_testy)

U_exact = exact_u(X_test, Y_test)

abs_err = np.abs(U_pred - U_exact)

# =====================================================
# Error norms
# =====================================================

rms_err = np.sqrt(np.mean(abs_err**2))

linf_err = np.max(abs_err)

print(
    f"Linf error = {linf_err:.2e}, "
    f"RMS error = {rms_err:.2e}"
)

# =====================================================
# 3D solution comparison
# =====================================================

fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111, projection='3d')

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

ax.set_xlabel("$x$", fontsize=12)

ax.set_ylabel("$y$", fontsize=12)

ax.set_zlabel("$u(x,y)$", fontsize=12)

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

plt.savefig("Figure3_a.pdf", bbox_inches="tight")

plt.show()

# =====================================================
# Absolute error contour plot
# =====================================================

plt.figure(figsize=(6, 5))

cp = plt.contourf(
    X_test,
    Y_test,
    abs_err,
    levels=40,
    cmap='viridis'
)

plt.colorbar(cp, label=r'$|u - \hat{u}|$')

plt.xlabel("$x$")

plt.ylabel("$y$")

plt.tight_layout()

plt.savefig("Figure3_b.pdf", bbox_inches="tight")

plt.show()

# =====================================================
# Residual vs Gauss Newton Iteration
# =====================================================

iters = np.arange(1, len(res_history) + 1)

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

plt.xlabel("Gauss--Newton iteration", fontsize=11)

plt.ylabel(r"$\|R\|_2$", fontsize=11)

plt.xticks(iters)

plt.grid()

plt.tight_layout()

plt.savefig("Figure3_r.pdf", bbox_inches="tight")

plt.show()
