import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre

# =====================================================
# Problem parameters
# =====================================================
a = np.e
b = 2*np.e
tol = 1e-12
maxit = 20

def u_exact(x):
    return np.log(np.log(x))

def x_to_t(x):
    return 2*(x - a)/(b - a) - 1

dt_dx = 2/(b - a)

# =====================================================
# Solver returning RMS error
# =====================================================
def solve_RMS_error(N, M):
    M_init = M

    # ---------------------------
    # Legendre basis
    # ---------------------------
    leg = [Legendre([0]*n + [1]) for n in range(N)]
    leg_d1 = [P.deriv() for P in leg]
    leg_d2 = [P.deriv(2) for P in leg]

    # ---------------------------
    # Initialization
    # ---------------------------
    x_init = np.linspace(a, b, M_init + 2)[1:-1]
    t_init = x_to_t(x_init)

    Phi2 = np.array([P(t_init) * dt_dx**2 for P in leg_d2])
    A_init = (x_init[:, None]**2) * Phi2.T
    rhs_init = -1 / np.log(x_init)

    t_bc = x_to_t(a)
    bc_vals  = np.array([P(t_bc) for P in leg])
    bc_d1val = np.array([P(t_bc) * dt_dx for P in leg_d1])

    A_init = np.vstack([A_init, bc_vals, bc_d1val])
    rhs_init = np.concatenate([rhs_init, [0.0, 1/np.e]])

    beta = np.linalg.lstsq(A_init, rhs_init, rcond=None)[0]

    # ---------------------------
    # Chebyshev collocation
    # ---------------------------
    k = np.arange(M + 1)
    x_cgl = np.cos(np.pi * k / M)
    x_col = a + (b - a) * (x_cgl[1:] + 1) / 2
    t_col = x_to_t(x_col)

    Phi1_c = np.array([P(t_col) * dt_dx for P in leg_d1])
    Phi2_c = np.array([P(t_col) * dt_dx**2 for P in leg_d2])

    # ---------------------------
    # Newton iteration
    # ---------------------------
    for _ in range(maxit):
        u_prime = beta @ Phi1_c
        u_pp    = beta @ Phi2_c

        F = np.zeros(M + 2)
        J = np.zeros((M + 2, N))

        F[:M] = (x_col**2) * u_pp + (x_col * u_prime)**2 + 1 / np.log(x_col)

        for i, x in enumerate(x_col):
            J[i, :] = (
                x**2 * Phi2_c[:, i]
                + 2 * (x * u_prime[i]) * x * Phi1_c[:, i]
            )

        # Boundary conditions
        F[M]   = beta @ bc_vals
        J[M,:] = bc_vals
        F[M+1] = beta @ bc_d1val - 1/np.e
        J[M+1,:] = bc_d1val

        if np.linalg.norm(F) < tol:
            break

        beta += np.linalg.lstsq(J, -F, rcond=None)[0]

    # ---------------------------
    # RMS error
    # ---------------------------
    x_test = np.linspace(a, b, 50)
    t_test = x_to_t(x_test)
    Phi_test = np.array([P(t_test) for P in leg])

    u_approx = beta @ Phi_test
    u_ex = u_exact(x_test)

    RMS_err = np.sqrt(np.mean((u_ex - u_approx)**2))
    return RMS_err


# =====================================================
# Convergence study: M vs RMS error
# =====================================================
N_list = [22, 24, 26, 28, 30]
M_list = [30, 35, 40, 45, 50]
colors = plt.cm.tab10.colors

plt.figure(figsize=(5.2, 4.2))

for i, N in enumerate(N_list):
    errors = []
    for M in M_list:
        err = solve_RMS_error(N, M)
        errors.append(err)

    plt.semilogy(
        M_list,
        errors,
        linestyle='-',
        marker='o',
        markersize=4.5,
        markeredgewidth=1.5,
        markeredgecolor='black',
        linewidth=1,
        color=colors[i % len(colors)],
        label=f"$N = {N}$"
    )

plt.xlabel("Number of collocation points")
plt.ylabel("RMS error")

plt.xticks(M_list)
plt.ylim(1e-18, 1e-12)

plt.legend(
    title="Basis functions",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="best"
)

plt.grid()
plt.tight_layout()

plt.savefig("Figure1_b.pdf", bbox_inches="tight")
plt.show()

