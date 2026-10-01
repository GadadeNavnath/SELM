#### Problem 1: Nonlinear Differential Equation

We consider the nonlinear differential equation:

$$
x^2 u''(x) + \left(x u'(x)\right)^2 + \frac{1}{\log x} = 0
$$

subject to the initial conditions:

$$
u(e) = 0, \quad u'(e) = \frac{1}{e}
$$

The exact solution is given by:

$$
u(x) = \log\big(\log x\big)
$$

---

#### Files

* `main.py`
  Solves the nonlinear initial value problem using the proposed Legendre-MELM method, computes the numerical solution, evaluates L∞ and RMS errors, and generates error and residual plots.

* `main1.py`
  Solves the same nonlinear initial value problem using PIELM for comparison and computes the corresponding numerical errors.

* `RMS_error.py`
  Generates a plot of RMS error versus the number of collocation points for varying numbers of basis functions.

#### How to run

1. Run `main.py` to reproduce `Figure1_a` and `Figure1_r` for Legendre-MELM.
2. Run `main1.py` to obtain the PIELM results for comparison.
3. Run `RMS_error.py` to reproduce `Figure1_b`.

#### Output

The code produces:

* `Figure1_a.pdf` (used in paper)
* `Figure1_b.pdf` (used in paper)
* `Figure1_r.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.
