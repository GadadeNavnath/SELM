#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
from scipy.linalg import lstsq
import time

#========================================================
# Initial Parameters
#========================================================
N = 26         # number of Legendre basis functions
M = 35        # collocation points for Newton iteration
M_init = 35    # collocation points for initialization
a = np.e
b = 2*np.e      # interval [e, 2e]
tol = 1e-12
maxit = 20
damping = 1.0

# Exact solution
def u_exact(x):
    return np.log(np.log(x))

# Mapping: x ∈ [a,b] → t ∈ [-1,1]
def x_to_t(x):
    return 2*(x - a)/(b - a) - 1
dt_dx = 2/(b - a)

#========================================================
# Legendre basis and its derivatives
#========================================================
leg = [Legendre([0]*n + [1]) for n in range(N)]
leg_d1 = [P.deriv() for P in leg]
leg_d2 = [P.deriv(2) for P in leg]

#========================================================
# Step 1 (Initialization--linearized system)
#========================================================
x_init = np.linspace(a, b, M_init+2)[1:-1] # interior points
t_init = x_to_t(x_init)

# Evaluate basis and derivatives at initialization points
Phi  = np.array([P(t_init) for P in leg])              # P(x)
Phi1 = np.array([P(t_init)*dt_dx for P in leg_d1])     # P'(x)
Phi2 = np.array([P(t_init)*dt_dx**2 for P in leg_d2])  # P''(x)

# Linearized equation: x²u'' + 1/log(x) = 0
A_init = (x_init[:, None]**2) * Phi2.T
rhs_init = -1/np.log(x_init)

# Initial conditions at x=a
t_bc = x_to_t(a)
bc_vals  = np.array([P(t_bc) for P in leg])
bc_d1val = np.array([P(t_bc)*dt_dx for P in leg_d1])

A_init = np.vstack([A_init, bc_vals, bc_d1val])
rhs_init = np.concatenate([rhs_init, [0.0, 1/np.e]])

# Solve for initial beta
t0 = time.time()
beta_init = lstsq(A_init, rhs_init, lapack_driver='gelsy')[0]
t_initial = time.time() - t0
print(f"Initialization time = {t_initial:.5f} s")
print("")

#========================================================
# Step 2 (Gauss - Newton Iteration)
#========================================================
k = np.arange(M+2)
x_cgl = np.cos(np.pi * k / (M+1))

x_cheb = a + (b - a) * (x_cgl + 1) / 2

x_col = x_cheb[1:-1]
t_col = x_to_t(x_col)

Phi_c  = np.array([P(t_col) for P in leg])
Phi1_c = np.array([P(t_col)*dt_dx for P in leg_d1])
Phi2_c = np.array([P(t_col)*dt_dx**2 for P in leg_d2])

#========================================================
# Residual and Jacobian
#========================================================
def residual_and_jacobian(beta):
    uprime = beta @ Phi1_c
    upp    = beta @ Phi2_c
    F = np.zeros(M+2)
    J = np.zeros((M+2, N))

    F[:M] = (x_col**2)*upp + (x_col*uprime)**2 + 1/np.log(x_col)
    J[:M, :] = ((x_col**2)[:, None] * Phi2_c.T +
                (2 * (x_col*uprime) * x_col)[:, None] * Phi1_c.T)

    F[M]   = beta @ bc_vals
    J[M,:] = bc_vals
    F[M+1]   = beta @ bc_d1val - 1/np.e
    J[M+1,:] = bc_d1val

    return F, J
#========================================================
# Newton iteration
#========================================================
beta = beta_init.copy()
res_history = []

t_start = time.time()

for k in range(maxit):
    F, J = residual_and_jacobian(beta)
    Fnrm = np.linalg.norm(F)
    res_history.append(Fnrm)

    print(f"Newton iter {k+1:2d}: residual = {Fnrm:.2e}")

    # QR-based least-squares solution
    delta = lstsq(J, -F, lapack_driver='gelsy')[0]

    # Store old beta
    beta_old = beta.copy()

    # Damped Gauss-Newton update
    beta += damping * delta

    # Relative change in beta
    rel_change = np.linalg.norm(beta - beta_old) / np.linalg.norm(beta_old)

    # Stopping criterion
    if rel_change < tol:
        break

t_newton = time.time() - t_start

print(f"\nNewton time = {t_newton:.4f} s")
print(f"Total time  = {t_initial + t_newton:.4f} s")
#========================================================
# Postprocessing
#========================================================
x_test = np.linspace(a, b, 200)
t_test = x_to_t(x_test)
Phi_test = np.array([P(t_test) for P in leg])

u_approx = beta @ Phi_test
u_ex = u_exact(x_test)

abs_err = np.abs(u_ex - u_approx)

RMS_err = np.sqrt(np.mean(abs_err**2))
print("")
print(f"Linf error = {np.max(abs_err):.2e}, RMS error = {RMS_err:.2e}")

plt.figure(figsize=(6, 5))
plt.plot(x_test, abs_err, "k-", linewidth=1.5)
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.grid(True)

plt.tight_layout()
plt.savefig("Figure1_a.pdf", bbox_inches="tight")
plt.show()

#========================================================
# Residual vs Gauss Newton Iteration
#========================================================

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

plt.xlabel("Gauss-Newton iteration", fontsize=11)
plt.ylabel(r"$\|R\|_2$", fontsize=11)

plt.xticks(iters)
plt.grid()
plt.tight_layout()
plt.savefig("Figure1_r.pdf", bbox_inches="tight")
plt.show()

