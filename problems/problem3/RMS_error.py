import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval, legder

# =====================================================
# Exact solution and right-hand side
# =====================================================

def exact_u(x, y):

    return (y**2) * np.sin(np.pi * x)

def exact_dudy_top(x):

    return 2 * np.sin(np.pi * x)

def rhs_func(x, y):

    return (
        np.sin(np.pi * x)
        * (
            2
            - (np.pi**2) * (y**2)
            + 2 * (y**3) * np.sin(np.pi * x)
        )
    )

# =====================================================
# Mapping utilities
# =====================================================

def to_legendre_domain(z):

    return 2*z - 1

# =====================================================
# 2D Legendre basis matrices
# =====================================================

def legendre_matrix_2d(x, y, N):

    xi = to_legendre_domain(x)
    eta = to_legendre_domain(y)

    cols = []

    for i in range(N):

        Px = legval(xi, [0]*i + [1])

        for j in range(N):

            Py = legval(eta, [0]*j + [1])

            cols.append(Px * Py)

    return np.vstack(cols).T

def legendre_derivative_matrices_2d(x, y, N):

    xi = to_legendre_domain(x)
    eta = to_legendre_domain(y)

    dx_cols = []
    dy_cols = []

    dxx_cols = []
    dyy_cols = []

    for i in range(N):

        ci = [0]*i + [1]

        d1i = legder(ci)
        d2i = legder(legder(ci))

        Pi = legval(xi, ci)
        dPi = legval(xi, d1i)
        d2Pi = legval(xi, d2i)

        for j in range(N):

            cj = [0]*j + [1]

            d1j = legder(cj)
            d2j = legder(legder(cj))

            Pj = legval(eta, cj)
            dPj = legval(eta, d1j)
            d2Pj = legval(eta, d2j)

            dx_cols.append(2 * dPi * Pj)
            dy_cols.append(2 * Pi * dPj)

            dxx_cols.append(4 * d2Pi * Pj)
            dyy_cols.append(4 * Pi * d2Pj)

    return (
        np.vstack(dx_cols).T,
        np.vstack(dy_cols).T,
        np.vstack(dxx_cols).T,
        np.vstack(dyy_cols).T
    )

# =====================================================
# Solver returning RMS error
# =====================================================

def solve_RMS_error(N_basis, Nxy, maxit=5, tol=1e-10):

    # -------------------------------------------------
    # Chebyshev--Gauss--Lobatto points
    # -------------------------------------------------

    k = np.arange(Nxy)

    x = 0.5 * (
        np.cos(np.pi * k / (Nxy - 1)) + 1
    )

    y = x.copy()

    X, Y = np.meshgrid(x, y)

    Xf = X.flatten()
    Yf = Y.flatten()

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

    # -------------------------------------------------
    # Basis matrices
    # -------------------------------------------------

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

    # -------------------------------------------------
    # RHS and boundary conditions
    # -------------------------------------------------

    RHS_int = rhs_func(X_int, Y_int)

    U_b_exact = exact_u(X_b, Y_b)

    mask_top = np.isclose(Y_b, 1.0)

    mask_other = ~mask_top

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------

    A_init = np.vstack([
        dxxP_int + dyyP_int,
        P_b[mask_other],
        dyP_b[mask_top]
    ])

    rhs_init = np.concatenate([
        RHS_int,
        U_b_exact[mask_other],
        exact_dudy_top(X_b[mask_top])
    ])

    beta = np.linalg.lstsq(
        A_init,
        rhs_init,
        rcond=None
    )[0]

    # -------------------------------------------------
    # Gauss--Newton iteration
    # -------------------------------------------------

    for _ in range(maxit):

        u = P_int @ beta

        uy = dyP_int @ beta

        uxx = dxxP_int @ beta

        uyy = dyyP_int @ beta

        R_int = (
            uxx
            + uyy
            + u * uy
            - RHS_int
        )

        u_b = P_b @ beta

        uy_b = dyP_b @ beta

        R = np.concatenate([
            R_int,
            u_b[mask_other] - U_b_exact[mask_other],
            uy_b[mask_top] - exact_dudy_top(X_b[mask_top])
        ])

        if np.linalg.norm(R) < tol:

            break

        J_nl = (
            np.diag(uy) @ P_int
            + np.diag(u) @ dyP_int
        )

        J_int = (
            dxxP_int
            + dyyP_int
            + J_nl
        )

        J = np.vstack([
            J_int,
            P_b[mask_other],
            dyP_b[mask_top]
        ])

        beta += np.linalg.lstsq(
            J,
            -R,
            rcond=None
        )[0]

    # -------------------------------------------------
    # RMS error on test grid
    # -------------------------------------------------

    Nt = 50

    xt = 0.5 * (
        np.cos(
            np.pi * np.arange(Nt) / (Nt - 1)
        ) + 1
    )

    yt = xt

    X_test, Y_test = np.meshgrid(xt, yt)

    U_pred = (
        legendre_matrix_2d(
            X_test.flatten(),
            Y_test.flatten(),
            N_basis
        )
        @ beta
    )

    U_exact = exact_u(
        X_test,
        Y_test
    ).flatten()

    rms_err = np.sqrt(
        np.mean((U_pred - U_exact)**2)
    )

    return rms_err

# =====================================================
# Convergence study
# =====================================================

N_list = [13, 16, 19, 22, 25]
M_list = [15, 20, 25, 30, 35]

colors = plt.cm.tab10.colors

plt.figure(figsize=(5.2, 4.2))

for i, N in enumerate(N_list):

    collocation_counts = []

    errors = []

    for M in M_list:

        Nc = M * M

        err = solve_RMS_error(N, M)

        collocation_counts.append(Nc)

        errors.append(err)

    plt.semilogy(
        collocation_counts,
        errors,
        linestyle='-',
        marker='o',
        markersize=4.5,
        markeredgewidth=1.3,
        markeredgecolor='black',
        linewidth=1.2,
        color=colors[i % len(colors)],
        label=fr"$N = {N}$"
    )

plt.xlabel("Total number of collocation points")

plt.ylabel("RMS error")

plt.legend(
    title="Basis functions per direction",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="best"
)

plt.grid()

plt.tight_layout()

plt.savefig("Figure3_c.pdf", bbox_inches="tight")

plt.show()
