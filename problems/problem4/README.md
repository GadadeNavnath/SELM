#### Problem 4: Elliptic Partial Differential Equation on an Irregular Domain

We consider the elliptic partial differential equation:

$$u_{xx} + u_{yy} = e^{-x}(x - 2 + y^3 + 6y), $$

over an irregular physical domain corresponding to the Chhattisgarh state boundary. 
The exact solution is given by: $$u(x,y)=e^{-x}(x+y^3).$$

---
#### Files

##### `main.py`

Solves the elliptic partial differential equation on the irregular domain, computes the numerical solution, evaluates the $$L_\infty$$ and RMS test errors, and test plots.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions.

---

#### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

#### `domain.py`

Constructs the irregular physical domain corresponding to the Chhattisgarh state geometry.

---

#### `spectral_convergence.py`

Constructs a plot of RMS error versus the number of basis functions per direction and demonstrates spectral (exponential) convergence.

---

#### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

---

#### How to run

Run `domain.py` to reproduce:

- `Figure4_a.pdf`

Run `main.py` to reproduce:

- `Figure4_b.pdf`

Run `RMS_error.py` to reproduce:

- `Figure4_c.pdf`

Run `spectral_convergence.py` to reproduce:

- `Figure4_s.pdf`

---

#### Output

The code produces:

- `Figure4_a.pdf` (used in paper)
- `Figure4_b.pdf` (used in paper)
- `Figure4_c.pdf` (used in paper)
- `Figure4_s.pdf` (used in paper)

Figures are saved in the `Figures/` folder.

---

#### Data

The `data/` folder contains:
- trainging collocation points,
- test collocation points,
- datasets used for RMS error plot.

All datasets are stored in NumPy `.npy` format.
