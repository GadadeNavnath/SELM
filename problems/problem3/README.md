#### Problem 3: Nonlinear Partial Differential Equation

We consider the nonlinear partial differential equation:

$$
u_{xx} + u_{yy} + uu_y = \sin(\pi x)\left(2 - \pi^2 y^2 + 2y^3\sin(\pi x)\right), \qquad (x,y)\in[0,1]^2
$$

The exact solution is given by:

$$
u(x,y) = y^2 \sin(\pi x).
$$

The boundary conditions are:

$$
u(x,0)=0,
\qquad
u(0,y)=0,
\qquad
u(1,y)=0,
$$

and

$$
u_y(x,1)=2\sin(\pi x).
$$

---

#### Files

* `main.py`  
  Solves the nonlinear partial differential equation, computes the numerical solution, evaluates the $L_\infty$ and RMS errors, and generates solution, error, and residual plots.

* `RMS_error.py`  
  Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions.

---

#### How to run

1. Run `main.py` to reproduce `Figure3_a`, `Figure3_b`, and `Figure3_r`.
2. Run `RMS_error.py` to reproduce `Figure3_c`.

---

#### Output

The code produces:

* `Figure3_a.pdf` (used in paper)
* `Figure3_b.pdf` (used in paper)
* `Figure3_c.pdf` (used in paper)
* `Figure3_r.pdf` (used in paper)

Figures are saved in the `Figures/` folder.

