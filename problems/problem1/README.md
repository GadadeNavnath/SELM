## Problem 1: Nonlinear Differential Equation

Solve the following nonlinear differential equation:

$$
x^2 u''(x) + \left(x u'(x)\right)^2 + \frac{1}{\log x} = 0
$$

subject to the initial conditions:

$$
u(e) = 0, \quad u'(e) = \frac{1}{e}
$$

### Analytical Solution

The exact solution is given by:

$$
u(x) = \log\big(\log x\big)
$$



# Problem 1: Nonlinear BVP (Spectral ELM)

## Files

- main_solver.ipynb  
  Solves the nonlinear boundary value problem using spectral basis and Gauss–Newton iteration.

- convergence_study.ipynb  
  Computes RMS error vs number of collocation points.

## How to run

1. Run `main_solver.ipynb` to compute solution.
2. Run `convergence_study.ipynb` to reproduce Figure 1.

## Output

- Figure1_b.pdf (used in paper)
- Figure1_b.png (for preview)
