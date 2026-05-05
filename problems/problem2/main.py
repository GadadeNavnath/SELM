import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
import time

# ======================================================
# Initial Parameters
# ======================================================
N = 21  # number of Legendre basis functions per variable
M = 60        # collocation points for Newton iteration
M_init = 60  # collocation points for initialization
a, b = 0.0, 3.0
tol = 1e-12
maxit = 20

# Exact solutions
def u1_exact(x): return np.sin(x)
def u2_exact(x): return 1 + x**2

# Mapping: x ∈ [a,b] → t ∈ [-1,1]
def x_to_t(x):
    return 2*(x - a)/(b - a) - 1
dt_dx = 2/(b - a)

# ======================================================
# Legendre basis and its derivatives
# ======================================================
leg = [Legendre([0]*n + [1]) for n in range(N)]
leg_d1 = [P.deriv() for P in leg]

# ======================================================
# Initialization step (linearized system)
# ======================================================
k = np.arange(0, M_init + 1)
x_cgl = np.cos(np.pi * k / M_init)

x_mapped = a + (b - a) * (x_cgl + 1) / 2
x_init = x_mapped[1:]
t_init = x_to_t(x_init)

Phi  = np.array([P(t_init) for P in leg])
Phi1 = np.array([P(t_init)*dt_dx for P in leg_d1])

S = 1 + x_init**2 + np.sin(x_init)**2
A_init = np.zeros((2*M_init + 2, 2*N))
rhs_init = np.zeros(2*M_init + 2)

for i, x in enumerate(x_init):
    A_init[2*i, :N]   = Phi1[:, i]
    A_init[2*i, N:]   = -Phi[:, i]
    rhs_init[2*i]     = np.cos(x) - S[i]

    A_init[2*i+1, N:] = Phi1[:, i]
    rhs_init[2*i+1]   = 2*x - (1+x**2)*np.sin(x)

# Boundary conditions
t_bc = x_to_t(a)
bc_vals = np.array([P(t_bc) for P in leg])

A_init[2*M_init, :N] = bc_vals       # u1(0) = 0
rhs_init[2*M_init]   = 0.0
A_init[2*M_init+1, N:] = bc_vals     # u2(0) = 1
rhs_init[2*M_init+1]   = 1.0

t0 = time.time()
beta_init = np.linalg.lstsq(A_init, rhs_init, rcond=None)[0]
t_initial = time.time() - t0
print(f"Initialization time = {t_initial:.5f} s\n")

# ======================================================
# Precompute for Newton iteration
# ======================================================
k = np.arange(M+1)
x_cgl = np.cos(np.pi * k / M)

x_cheb = a + (b - a) * (x_cgl + 1) / 2
x_col = x_cheb[1:]
t_col = x_to_t(x_col)

Phi_c  = np.array([P(t_col) for P in leg])
Phi1_c = np.array([P(t_col)*dt_dx for P in leg_d1])

# ======================================================
# Residual and Jacobian
# ======================================================
def residual_and_jacobian(beta):
    a_coef = beta[:N]
    b_coef = beta[N:]

    u1 = a_coef @ Phi_c
    u1p = a_coef @ Phi1_c
    u2 = b_coef @ Phi_c
    u2p = b_coef @ Phi1_c

    F = np.zeros(2*M + 2)
    J = np.zeros((2*M + 2, 2*N))
    
    for i, x in enumerate(x_col):
        Sx = 1 + x**2 + np.sin(x)**2

        F[2*i]   = u1p[i] - (np.cos(x) + u1[i]**2 + u2[i] - Sx)
        F[2*i+1] = u2p[i] - (2*x - (1+x**2)*np.sin(x) + u1[i]*u2[i])

        J[2*i, :N]   = Phi1_c[:, i] - 2*u1[i]*Phi_c[:, i]
        J[2*i, N:]   = -Phi_c[:, i]
        J[2*i+1, :N] = -u2[i]*Phi_c[:, i]
        J[2*i+1, N:] = Phi1_c[:, i] - u1[i]*Phi_c[:, i]

    J[2*M, :N] = bc_vals;   F[2*M]   = a_coef @ bc_vals
    J[2*M+1, N:] = bc_vals; F[2*M+1] = b_coef @ bc_vals - 1.0
    return F, J

# ======================================================
# Newton iteration
# ======================================================
beta = beta_init.copy()
res_history = []

t_start = time.time()
for k in range(maxit):
    F, J = residual_and_jacobian(beta)
    Fnrm = np.linalg.norm(F)
    res_history.append(Fnrm)
    print(f"Newton iter {k+1:2d}: residual = {Fnrm:.3e}")
    if Fnrm < tol:
        break
    delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
    beta += delta
t_newton = time.time() - t_start

print(f"\nNewton time = {t_newton:.4f} s")
print(f"Total time  = {t_initial + t_newton:.4f} s")

# ======================================================
# Postprocessing
# ======================================================
x_test = np.linspace(a, b,200)
t_test = x_to_t(x_test)
Phi_test = np.array([P(t_test) for P in leg])

u1_approx = beta[:N] @ Phi_test
u2_approx = beta[N:] @ Phi_test
u1_ex = u1_exact(x_test)
u2_ex = u2_exact(x_test)

abs_err_u1 = np.abs(u1_ex - u1_approx)
abs_err_u2 = np.abs(u2_ex - u2_approx)

rms_err_u1 = np.sqrt(np.mean(abs_err_u1**2))
rms_err_u2 = np.sqrt(np.mean(abs_err_u2**2))

print("")
print(f"Linf error (u1) = {np.max(abs_err_u1):.2e}, RMS error = {rms_err_u1:.2e}")
print(f"Linf error (u2) = {np.max(abs_err_u2):.2e}, RMS error = {rms_err_u2:.2e}")

plt.figure(figsize=(6, 5))

plt.plot(x_test, abs_err_u1, color="black", linewidth=1.5, label=r"$|u_1 - \hat{u}_1|$")
plt.plot(x_test, abs_err_u2, color="blue", linewidth=1.5, label=r"$|u_2 - \hat{u}_2|$")

plt.xlabel("x", fontsize=12)
plt.ylabel("Absolute Error", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

plt.tight_layout()
plt.savefig("Figure2_a.pdf", bbox_inches="tight")
plt.show()

# ======================================================
# Residual vs Gauss Newton Iteration
# ======================================================
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
plt.savefig("Figure2_r.pdf", bbox_inches="tight")
plt.show()

