#### Problem 6: Nonlinear Elliptic Partial Differential Equation on a Star-Shaped Domain

We consider the nonlinear elliptic partial differential equation

$$
-(u_{xx}+u_{yy})+u^3 = 2\pi^2\sin(\pi x)\sin(\pi y) + \left(\sin(\pi x)\sin(\pi y)\right)^3, $$

over a star-shaped irregular domain.

The exact solution is given by

$$
u(x,y)=\sin(\pi x)\sin(\pi y).
$$

---

#### Files

##### `domain.py`

Constructs the star-shaped irregular domain, generates the training collocation datasets, and reproduces:

- `Figure6_a.pdf`

The generated training datasets are stored in the `data/` folder.

---

##### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

The generated test datasets are stored in the `data/` folder.

---

##### `main.py`

Solves the nonlinear elliptic partial differential equation on the star-shaped domain, computes the numerical solution, evaluates the $L_\infty$ and RMS errors, and reproduces:

- `Figure6_b.pdf`
- `Figure6_r.pdf`

This script loads the training and test datasets from the `data/` folder.

---

##### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions, and reproduces:

- `Figure6_c.pdf`

This script loads the RMS error datasets from the `data/` folder.

---

#### Recommended execution order

1. Run `domain.py`
2. Run `test_data_creation.py`
3. Run `RMS_error_data_creation.py`
4. Run `main.py`
5. Run `RMS_error.py`

---

#### Output

The code produces:

- `Figure6_a.pdf` (used in paper)
- `Figure6_b.pdf` (used in paper)
- `Figure6_c.pdf` (used in paper)
- `Figure6_r.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.

---

#### Data

The `data/` folder contains:

- training collocation points,
- test collocation points,
- datasets used for the RMS error study.

All datasets are stored in NumPy `.npy` format.
