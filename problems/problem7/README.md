#### Problem 7: Three-Dimensional Nonlinear Partial Differential Equation on a Spherical Shell Domain

We consider the three-dimensional nonlinear partial differential equation

$$
u_{xx} + u_{yy} + u_{zz} - u^2 = (2 + y^2)e^x - z^2 \sin(y) -\left[e^x y^2 + (z^2 + 2)\sin(y)\right]^2, $$

on a spherical shell domain.

The domain is most conveniently described in spherical coordinates
$(r,\theta,\phi)$ as

$$
r \in [0.5,1],
\qquad
\theta \in [0,\pi/2],
\qquad
\phi \in [0,\pi/2].
$$

However, the problem is solved using Cartesian coordinates
$(x,y,z)$.

The exact solution is given by

$$
u(x,y,z)
=
e^x y^2 + (z^2 + 2)\sin(y).
$$

---

#### Files

##### `domain.py`

Constructs the spherical shell domain, generates the training collocation datasets, and reproduces:

- `Figure7_a.pdf`

The generated training datasets are stored in the `data/` folder.

---

##### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

The generated test datasets are stored in the `data/` folder.

---

##### `main.py`

Solves the three-dimensional nonlinear partial differential equation on the spherical shell domain, computes the numerical solution, evaluates the $L_\infty$ and RMS errors, and reproduces:

- `Figure7_b.pdf`
- `Figure7_r.pdf`

This script loads the training and test datasets from the `data/` folder.

---

##### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions, and reproduces:

- `Figure7_c.pdf`

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

- `Figure7_a.pdf` (used in paper)
- `Figure7_b.pdf` (used in paper)
- `Figure7_c.pdf` (used in paper)
- `Figure7_r.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.

---

#### Data

The `data/` folder contains:

- training collocation points,
- test collocation points,
- datasets used for the RMS error study.

All datasets are stored in NumPy `.npy` format.
