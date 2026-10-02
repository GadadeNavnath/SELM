#### Porblem 7: We consider the nonlinear Bratu problem:

$$
u_{xx} + u_{yy} + \lambda e^u = 0,
\qquad (x,y)\in[0,1]^2,
$$

where

$$
\lambda = 6.808124426.
$$

Homogeneous Dirichlet boundary conditions are prescribed along the boundary:

$$
u(x,y)=0.
$$

The value of $\lambda$ corresponds to the critical parameter of the two-dimensional Bratu problem.

---

#### Files

* `main.ipynb`  
  Solves the nonlinear Bratu problem using the Legendre-MELM method, computes the numerical solution, and generates the solution, error, and residual plots.

* `main1.ipynb`  
  Computes the PIELM results for comparison.

* `P2_FEM.ipynb`  
  Computes the FEM reference solution for comparison.

* `fem.csv`  
  Contains the FEM reference data used for numerical comparison.

---

#### How to run

1. Run `main.ipynb` to reproduce `Figure7_a`, `Figure7_b`, and `Figure7_r`.
2. Run `main1.ipynb` to obtain the PIELM results.
3. Run `P2_FEM.ipynb` to generate the FEM reference solution, if required.

---

#### Output

The code produces:

* `Figure7_a.pdf` (used in paper)
* `Figure7_b.pdf` (used in paper)
* `Figure7_r.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.
"""
