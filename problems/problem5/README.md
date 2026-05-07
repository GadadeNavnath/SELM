#### Problem 5: Nonlinear Elliptic Partial Differential Equation on an Irregular Domain

We consider the nonlinear elliptic partial differential equation

$$
u_{xx} + u_{yy} + e^{u} = 1 + x^2 + y^2 + \frac{4}{(1+x^2+y^2)^2},$$

over an irregular physical domain corresponding to the Haryana state boundary.

The exact solution is given by

$$
u(x,y)=\log(1+x^2+y^2).
$$

---

#### Files

##### `domain.py`

Constructs the irregular physical domain corresponding to the Haryana state geometry, generates the training collocation datasets, and reproduces:

- `Figure5_a.pdf`

The generated training datasets are stored in the `data/` folder.

---

##### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

The generated test datasets are stored in the `data/` folder.

---

##### `main.py`

Solves the nonlinear elliptic partial differential equation on the irregular domain, computes the numerical solution, evaluates the $L_\infty$ and RMS errors, and reproduces:

- `Figure5_b.pdf`
- `Figure5_r.pdf`

This script loads the training and test datasets from the `data/` folder.

---

##### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions, and reproduces:

- `Figure5_c.pdf`

This script loads the RMS error datasets from the `data/` folder.

---

##### `spectral_convergence.py`

Constructs a plot of RMS error versus the number of basis functions per direction and demonstrates spectral (exponential) convergence.

Reproduces:

- `Figure5_s.pdf`

This script loads the training datasets from the `data/` folder.

#### Recommended execution order

1. Run `domain.py`
2. Run `test_data_creation.py`
3. Run `RMS_error_data_creation.py`
4. Run `main.py`
5. Run `RMS_error.py`
6. Run `spectral_convergence.py`

---

#### Output

The code produces:

- `Figure5_a.pdf` (used in paper)
- `Figure5_b.pdf` (used in paper)
- `Figure5_c.pdf` (used in paper)
- `Figure5_r.pdf` (used in paper)
- `Figure5_s.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.

---

#### Data

The `data/` folder contains:

- training collocation points,
- test collocation points,
- datasets used for the RMS error study.

All datasets are stored in NumPy `.npy` format.
