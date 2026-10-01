#### Problem 4: Elliptic Partial Differential Equation on an Irregular Domain

We consider the elliptic partial differential equation

$$
u_{xx} + u_{yy} = e^{-x}(x - 2 + y^3 + 6y),
$$

over an irregular physical domain corresponding to the Chhattisgarh state boundary.

The exact solution is given by

$$
u(x,y)=e^{-x}(x+y^3).
$$

---

#### Files

##### `domain.py`

Constructs the irregular physical domain corresponding to the Chhattisgarh state geometry, generates the training collocation datasets, and reproduces:

- `Figure4_a.pdf`

The generated training datasets are stored in the `data/` folder.

---

##### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

The generated test datasets are stored in the `data/` folder.

---

##### `main.py`

Solves the elliptic partial differential equation on the irregular domain using the Legendre-MELM method, computes the numerical solution, evaluates the $L_\infty$ and RMS test errors, and reproduces:

- `Figure4_b.pdf`

This script loads the training and test datasets from the `data/` folder.

##### `main1.py`

Computes the PIELM results for comparison using the training and test datasets from the `data/` folder.

---

##### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions, and reproduces:

- `Figure4_c.pdf`

This script loads the RMS error datasets from the `data/` folder.

---

##### `spectral_convergence.py`

Constructs a plot of RMS error versus the number of basis functions per direction and demonstrates spectral (exponential) convergence.

Reproduces:

- `Figure4_s.pdf`

This script loads the training datasets from the `data/` folder.

#### Recommended execution order

1. Run `domain.py`
2. Run `test_data_creation.py`
3. Run `RMS_error_data_creation.py`
4. Run `main.py`
5. Run `main1.py`
6. Run `RMS_error.py`
7. Run `spectral_convergence.py`

---

#### Output

The code produces:

- `Figure4_a.pdf` (used in paper)
- `Figure4_b.pdf` (used in paper)
- `Figure4_c.pdf` (used in paper)
- `Figure4_s.pdf` (used in paper)

Generated figures are provided in the `Figures/` folder.

---

#### Data

The `data/` folder contains:

- training collocation points,
- test collocation points,
- datasets used for the RMS error study.

All datasets are stored in NumPy `.npy` format.
