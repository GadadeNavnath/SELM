### Spectral Extreme Learning Machine

This repository contains reproducibility materials for the paper:

**“Spectral Extreme Learning Machine”**

**Authors:** Gadade Navnath Ankush and Sivaram Ambikasaran

---

This repository provides a Python implementation for solving differential equations. The code is organized into seven benchmark problems, covering nonlinear ordinary differential equations, systems of nonlinear ODEs, and partial differential equations on both regular and irregular domains, including a three-dimensional case.

---

Requirements
The code is implemented in Python and requires the following packages:

Python 3.12+

numpy

matplotlib

Problems 4, 5, and 6 additionally require:

shapely

Problems 4 and 5 additionally require:

geopandas

Problem 7 additionally requires:

scipy

Install the dependencies using:

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

### Methodology

The solution of the differential equations is obtained using a two-step procedure based on the Legendre-IELM framework combined with a Gauss–Newton iteration. In the first step, an initial approximation is constructed by forming an overdetermined linear system using collocation points and basis functions, and solving it in the least-squares sense to obtain an initial coefficient vector. In the second step, the nonlinear problem is solved iteratively using a Gauss–Newton method: at each iteration, the residual and its Jacobian with respect to the coefficients are evaluated, a linear least-squares problem is solved to compute the update, and the coefficients are refined until the residual norm satisfies a prescribed tolerance or the maximum number of iterations is reached. For Problem 4 (linear PDE), the solution is obtained directly from the initial least-squares formulation, and no Gauss–Newton iteration is required.

---

### Data for Irregular Domains

For Problems 4 and 5, the physical domains correspond to the Indian states of **Chhattisgarh** and **Haryana**.

Preprocessed and rescaled geometry data is included in this repository to ensure full reproducibility without requiring external downloads or additional dependencies.

The original dataset is obtained from:
https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json

---

### Repository Structure

The repository is organized using a main folder named `problems`, which contains the following subfolders:

```text
problems/
├── problem1/
├── problem2/
├── problem3/
├── problem4/
├── problem5/
├── problem6/
└── problem7/
```

Each problem folder includes:

* Python scripts for implementation
* Generated figures (stored in the `Figures/` folder)
* A dedicated README file explaining the problem setup and usage

---

### Reproducibility

This repository provides all code and data required to reproduce the numerical results presented in the paper.

Each problem folder contains scripts and instructions to reproduce the corresponding figures. All parameter settings are defined within the scripts.

Running the provided codes will reproduce all figures and results reported in the paper.

---

### License

This project is licensed under the MIT License.

