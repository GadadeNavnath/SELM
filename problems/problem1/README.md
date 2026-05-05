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
#### Files

* `main.py`
Solves the nonlinear initial value problem and computes the numerical solution, along with L∞ and RMS errors, and generates error and residual plots.
* `RMS_error.py`
  Computes RMS error versus the number of collocation points for varying numbers of basis functions.

---

#### How to run

1. Run `main.py` to reproduce `Figure1_a` and `Figure1_r`.
2. Run `RMS_error.py` to reproduce `Figure1_b`.

---

#### Output

The code produces:

* `Figure1_a.pdf` (used in paper)
* `Figure1_b.pdf` (used in paper)
* `Figure1_r.pdf` (used in paper)
 Figures are saved in the `Figures/` folder.
