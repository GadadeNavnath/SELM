#### Problem 4: Elliptic Partial Differential Equation on an Irregular Domain

We consider the elliptic partial differential equation:

$$u_{xx} + u_{yy} = e^{-x}(x - 2 + y^3 + 6y), $$

over an irregular physical domain corresponding to the Chhattisgarh state boundary. 
The exact solution is given by: $$u(x,y)=e^{-x}(x+y^3).$$

---
#### Files

##### `main.py`

Solves the elliptic partial differential equation on the irregular domain, computes the numerical solution, evaluates the $$L_\infty$$ and RMS errors, and generates solution, error, and residual plots.

---

##### `RMS_error.py`

Generates a plot of RMS error versus the total number of collocation points for varying numbers of basis functions.

---

#### `RMS_error_data_creation.py`

Generates multiple interior and boundary collocation datasets used in the RMS error study.

The generated datasets are stored in the `data/` folder.

---

#### `domain.py`

Constructs the irregular computational domain corresponding to the Chhattisgarh state geometry.

---

#### `spectral_convergence.py`

Performs the spectral convergence study with increasing numbers of basis functions.

---

#### `test_data_creation.py`

Generates additional test collocation datasets used for numerical validation.

---

#### How to run

Run `main.py` to reproduce:

- `Figure4_a.pdf`
- `Figure4_b.pdf`
- `Figure4_s.pdf`

Run `RMS_error.py` to reproduce:

- `Figure4_c.pdf`

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
- interior collocation points,
- boundary collocation points,
- datasets used for RMS error and convergence studies.

All datasets are stored in NumPy `.npy` format.

---

##### Python Requirements

The codes were tested using Python 3.

Required libraries:

- numpy
- matplotlib
- geopandas
- shapely

Install dependencies using:

```bash
pip install numpy matplotlib geopandas shapely




