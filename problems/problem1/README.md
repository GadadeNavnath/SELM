Problem 1. Solve the following non linear differential equation.

$$x^2 y'' + (xy')^2 +1 /(logx) = 0$$
 
$y(e) = 0$  and $y'(e) =1/e$

The analytical solution is :  $y(x) = \log(\log(x))$




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
