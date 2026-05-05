# Spectral Extreme Learning Machine (SELM)

This repository contains a Python implementation of the Spectral Extreme Learning Machine (SELM) for solving nonlinear differential equations. The code is organized into seven benchmark problems, covering nonlinear ordinary differential equations, systems of nonlinear ODEs, and nonlinear partial differential equations on both regular and irregular domains, including a three-dimensional case.

---

## Problem Overview

| Problem   | Type           | Equation Class | Domain    |
| --------- | -------------- | -------------- | --------- |
| Problem 1 | Single ODE     | Nonlinear      | Regular   |
| Problem 2 | System of ODEs | Nonlinear      | Regular   |
| Problem 3 | PDE            | Nonlinear      | Regular   |
| Problem 4 | PDE            | Linear         | Irregular |
| Problem 5 | PDE            | Nonlinear      | Irregular |
| Problem 6 | PDE            | Nonlinear      | Irregular |
| Problem 7 | PDE (3D)       | Nonlinear      | Irregular |

---

## Methodology

The solution of the differential equations is obtained using a two-step procedure based on the Legendre-IELM framework combined with a Gauss–Newton iteration. In the first step, an initial approximation is constructed by forming an overdetermined linear system using collocation points and basis functions, and solving it in the least-squares sense to obtain an initial coefficient vector. In the second step, the nonlinear problem is solved iteratively using a Gauss–Newton method: at each iteration, the residual and its Jacobian with respect to the coefficients are evaluated, a linear least-squares problem is solved to compute the update, and the coefficients are refined until the residual norm satisfies a prescribed tolerance or the maximum number of iterations is reached.

Each problem includes a separate README file describing the corresponding differential equation and implementation details.

---

## Data for Irregular Domains

For the highly irregular domains considered in Problems 4 and 5, geometric data is obtained from the GADM dataset using the following source:

```python
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json"
gdf = gpd.read_file(url)

geom_chhattisgarh = gdf[gdf["NAME_1"] == "Chhattisgarh"].geometry.values[0]
geom_haryana = gdf[gdf["NAME_1"] == "Haryana"].geometry.values[0]
```

The geometries corresponding to the Indian states of **Chhattisgarh** and **Haryana** are extracted, rescaled to the computational domain, and stored for use in the numerical implementation. Preprocessed geometry data is stored locally to ensure reproducibility and avoid repeated downloads.
## Repository Structure

The repository is organized problem-wise, with each folder containing the corresponding implementation and documentation:

```
Problem1/
Problem2/
Problem3/
Problem4/
Problem5/
Problem6/
Problem7/
```

Each problem folder includes:

* Python scripts for implementation
* Generated figures (stored in the `Figures/` folder)
* A dedicated README file explaining the problem setup and usage
