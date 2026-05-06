#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre

# =====================================================
# Problem parameters
# =====================================================
a, b = 0.0, 3.0
tol = 1e-12
maxit = 20

def u1_exact(x):
    return np.sin(x)

def u2_exact(x):
    return 1 + x**2

def x_to_t(x):
    return 2*(x - a)/(b - a) - 1

dt_dx = 2/(b - a)

# =====================================================
# Solver returning RMS errors for u1 and u2
# =====================================================
def solve_RMS_errors_u1_u2(N, M):
    """
    Solve the coupled nonlinear system and return
    RMS errors for u1 and u2
    """

    M_init = M

    # ---------------------------
    # Legendre basis
    # ---------------------------
    leg = [Legendre([0]*n + [1]) for n in range(N)]
    leg_d1 = [P.deriv() for P in leg]

    # ---------------------------
    # Initialization 
    # ---------------------------
    k = np.arange(0, M_init + 1)
    x_cgl = np.cos(np.pi * k / M_init)

    x_init = a + (b - a) * (x_cgl[1:] + 1) / 2
    t_init = x_to_t(x_init)

    Phi  = np.array([P(t_init) for P in leg])
    Phi1 = np.array([P(t_init)*dt_dx for P in leg_d1])

    A_init = np.zeros((2*M_init + 2, 2*N))
    rhs_init = np.zeros(2*M_init + 2)

    for i, x in enumerate(x_init):

        S = 1 + x**2 + np.sin(x)**2

        A_init[2*i, :N] = Phi1[:, i]
        A_init[2*i, N:] = -Phi[:, i]
        rhs_init[2*i] = np.cos(x) - S

        A_init[2*i+1, N:] = Phi1[:, i]
        rhs_init[2*i+1] = 2*x - (1+x**2)*np.sin(x)

    t_bc = x_to_t(a)
    bc_vals = np.array([P(t_bc) for P in leg])

    A_init[2*M_init, :N] = bc_vals
    rhs_init[2*M_init] = 0.0

    A_init[2*M_init+1, N:] = bc_vals
    rhs_init[2*M_init+1] = 1.0

    beta = np.linalg.lstsq(A_init, rhs_init, rcond=None)[0]

    # ---------------------------
    # Collocation points 
    # ---------------------------
    k = np.arange(M + 1)
    x_cgl = np.cos(np.pi * k / M)

    x_col = a + (b - a) * (x_cgl[1:] + 1) / 2
    t_col = x_to_t(x_col)

    Phi_c  = np.array([P(t_col) for P in leg])
    Phi1_c = np.array([P(t_col)*dt_dx for P in leg_d1])

    # ---------------------------
    # Newton iteration
    # ---------------------------
    for _ in range(maxit):

        a_coef = beta[:N]
        b_coef = beta[N:]

        u1  = a_coef @ Phi_c
        u1p = a_coef @ Phi1_c

        u2  = b_coef @ Phi_c
        u2p = b_coef @ Phi1_c

        F = np.zeros(2*M + 2)
        J = np.zeros((2*M + 2, 2*N))

        for i, x in enumerate(x_col):

            Sx = 1 + x**2 + np.sin(x)**2

            F[2*i] = (
                u1p[i]
                - (np.cos(x) + u1[i]**2 + u2[i] - Sx)
            )

            F[2*i+1] = (
                u2p[i]
                - (2*x - (1+x**2)*np.sin(x) + u1[i]*u2[i])
            )

            J[2*i, :N] = (
                Phi1_c[:, i]
                - 2*u1[i]*Phi_c[:, i]
            )

            J[2*i, N:] = -Phi_c[:, i]

            J[2*i+1, :N] = -u2[i]*Phi_c[:, i]

            J[2*i+1, N:] = (
                Phi1_c[:, i]
                - u1[i]*Phi_c[:, i]
            )

        J[2*M, :N] = bc_vals
        F[2*M] = a_coef @ bc_vals

        J[2*M+1, N:] = bc_vals
        F[2*M+1] = b_coef @ bc_vals - 1.0

        if np.linalg.norm(F) < tol:
            break

        beta += np.linalg.lstsq(J, -F, rcond=None)[0]

    # ---------------------------
    # RMS errors (independent grid)
    # ---------------------------
    x_test = np.linspace(a, b, 50)

    t_test = x_to_t(x_test)

    Phi_test = np.array([P(t_test) for P in leg])

    u1_approx = beta[:N] @ Phi_test
    u2_approx = beta[N:] @ Phi_test

    u1_ex = u1_exact(x_test)
    u2_ex = u2_exact(x_test)

    RMS_u1 = np.sqrt(np.mean((u1_ex - u1_approx)**2))
    RMS_u2 = np.sqrt(np.mean((u2_ex - u2_approx)**2))

    return RMS_u1, RMS_u2


# =====================================================
# Convergence study
# =====================================================
N_list = [20, 21, 22, 23, 24]
M_list = [40, 50, 60, 70, 80]

colors = plt.cm.tab10.colors

# =====================================================
# Plot for u1
# =====================================================
plt.figure(figsize=(5, 4))

for i, N in enumerate(N_list):

    errors_u1 = []

    for M in M_list:

        err_u1, _ = solve_RMS_errors_u1_u2(N, M)
        errors_u1.append(err_u1)

    plt.semilogy(
        M_list,
        errors_u1,
        linestyle='-',
        marker='o',
        markersize=4.5,
        markeredgewidth=1.5,
        markeredgecolor='black',
        linewidth=1,
        color=colors[i % len(colors)],
        label=f"$N = {N}$"
    )

plt.xlabel("Number of collocation points", fontsize=11)

plt.ylabel(r"RMS error of $u_1$", fontsize=11)

plt.xticks(M_list)

plt.ylim(1e-16, 1e-10)

plt.legend(
    title="Basis functions",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="best"
)

plt.grid()

plt.tight_layout()

plt.savefig("Figure2_b1.pdf", bbox_inches="tight")
plt.show()

# =====================================================
# Plot for u2
# =====================================================
plt.figure(figsize=(5.2, 4.2))

for i, N in enumerate(N_list):

    errors_u2 = []

    for M in M_list:

        _, err_u2 = solve_RMS_errors_u1_u2(N, M)
        errors_u2.append(err_u2)

    plt.semilogy(
        M_list,
        errors_u2,
        linestyle='-',
        marker='o',
        markersize=4.5,
        markeredgewidth=1.5,
        markeredgecolor='black',
        linewidth=1,
        color=colors[i % len(colors)],
        label=f"$N = {N}$"
    )

plt.xlabel("Number of collocation points", fontsize=11)

plt.ylabel(r"RMS error of $u_2$", fontsize=11)

plt.xticks(M_list)

plt.ylim(1e-16, 1e-10)

plt.legend(
    title="Basis functions",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
    loc="best"
)

plt.grid()

plt.tight_layout()

plt.savefig("Figure2_b2.pdf", bbox_inches="tight")
plt.show()


# In[ ]:




