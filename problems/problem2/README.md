#### Problem 2: System of coupled nonlinear ODEs

We consider the following system:

$$
u_1'(x) = \cos(x) + u_1^2(x) + u_2(x) - \left(1 + x^2 + \sin^2(x)\right)
$$

$$
u_2'(x) = 2x - (1+x^2)\sin(x) + u_1(x)u_2(x)
$$

subject to the initial conditions:

$$
u_1(0) = 0, \qquad u_2(0) = 1
$$

The exact solutions are given by:

$$
u_1(x) = \sin(x),
\qquad
u_2(x) = 1 + x^2
$$

---

#### Files

* `main.py`
  Solves the coupled nonlinear differential system using the Legendre-MELM method, computes the numerical solutions, evaluates the $L_\infty$ and RMS errors, and generates error and residual plots.

* `main1.py`
  Computes the PIELM results for comparison.

* `RMS_error.py`
  Generates plots of RMS error versus the number of collocation points for varying numbers of basis functions.

#### How to run

1. Run `main.py` to reproduce `Figure2_a` and `Figure2_r`.
2. Run `main1.py` to obtain the PIELM results.
3. Run `RMS_error.py` to reproduce `Figure2_b1` and `Figure2_b2`.
---

#### Output

The code produces:

* `Figure2_a.pdf` (used in paper)
* `Figure2_b1.pdf` (used in paper)
* `Figure2_b2.pdf` (used in paper)
* `Figure2_r.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.
