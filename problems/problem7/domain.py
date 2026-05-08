#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

# ======================================================
# Shell parameters
# ======================================================
R_inner = 0.5
R_outer = 1.0

# ======================================================
# Robust near-duplicate remover
# ======================================================
def remove_near_duplicates(points, tol=1e-12):

    tree = cKDTree(points)

    keep = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):

        if not keep[i]:
            continue

        idx = tree.query_ball_point(p, tol)

        idx.remove(i)

        keep[idx] = False

    return points[keep]

# ======================================================
# SAFE spherical points in POSITIVE OCTANT
# ======================================================
def spherical_layer_points_positive_octant(
    r,
    N_theta,
    N_phi,
    pole_tol=1e-10
):

    pts = []

    # --------------------------------------------------
    # Cell-centered sampling (prevents pole hits)
    # --------------------------------------------------
    theta = (
        np.arange(N_theta) + 0.5
    ) * (0.5 * np.pi / N_theta)

    phi = (
        np.arange(N_phi) + 0.5
    ) * (0.5 * np.pi / N_phi)

    for ph in phi:

        sin_ph = np.sin(ph)

        # ----------------------------------------------
        # Collapse near pole to single point
        # ----------------------------------------------
        if sin_ph < pole_tol:

            x = 0.0
            y = 0.0
            z = r * np.cos(ph)

            pts.append([x, y, z])

            continue

        for th in theta:

            x = r * sin_ph * np.cos(th)

            y = r * sin_ph * np.sin(th)

            z = r * np.cos(ph)

            pts.append([x, y, z])

    return pts

# ======================================================
# Radial boundary layers
# ======================================================
def radial_layers_outer(
    R_inner,
    R_outer,
    n_layers,
    base_spacing,
    growth,
    N_theta,
    N_phi
):

    pts = []

    d = base_spacing

    r = R_outer

    for _ in range(n_layers):

        r -= d

        if r <= R_inner:
            break

        pts += spherical_layer_points_positive_octant(
            r,
            N_theta,
            N_phi
        )

        d *= growth

    return pts, r


def radial_layers_inner(
    R_inner,
    R_outer,
    n_layers,
    base_spacing,
    growth,
    N_theta,
    N_phi
):

    pts = []

    d = base_spacing

    r = R_inner

    for _ in range(n_layers):

        r += d

        if r >= R_outer:
            break

        pts += spherical_layer_points_positive_octant(
            r,
            N_theta,
            N_phi
        )

        d *= growth

    return pts, r

# ======================================================
# Planar boundary layers
# ======================================================
def planar_boundary_layers(
    axis,
    n_layers,
    base_spacing,
    growth,
    N_plane
):

    pts = []

    d = base_spacing

    grid = np.linspace(0.0, R_outer, N_plane)

    for _ in range(n_layers):

        if axis == "x":

            for y in grid:
                for z in grid:

                    r = np.sqrt(d**2 + y**2 + z**2)

                    if R_inner <= r <= R_outer:

                        pts.append([d, y, z])

        elif axis == "y":

            for x in grid:
                for z in grid:

                    r = np.sqrt(x**2 + d**2 + z**2)

                    if R_inner <= r <= R_outer:

                        pts.append([x, d, z])

        elif axis == "z":

            for x in grid:
                for y in grid:

                    r = np.sqrt(x**2 + y**2 + d**2)

                    if R_inner <= r <= R_outer:

                        pts.append([x, y, d])

        d *= growth

    return pts

# ======================================================
# Interior core fill
# ======================================================
def core_fill(
    r_inner_core,
    r_outer_core,
    N_r,
    N_theta,
    N_phi
):

    pts = []

    radii = np.linspace(
        r_inner_core,
        r_outer_core,
        N_r
    )

    for r in radii:

        pts += spherical_layer_points_positive_octant(
            r,
            N_theta,
            N_phi
        )

    return pts

# ======================================================
# Generate INTERIOR points
# ======================================================
outer_pts, r_outer_core = radial_layers_outer(
    R_inner,
    R_outer,
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    N_theta=16,
    N_phi=16
)

inner_pts, r_inner_core = radial_layers_inner(
    R_inner,
    R_outer,
    n_layers=2,
    base_spacing=0.01,
    growth=1.3,
    N_theta=8,
    N_phi=8
)

plane_x_pts = planar_boundary_layers(
    "x",
    2,
    0.01,
    1.3,
    16
)

plane_y_pts = planar_boundary_layers(
    "y",
    2,
    0.01,
    1.3,
    16
)

plane_z_pts = planar_boundary_layers(
    "z",
    2,
    0.01,
    1.3,
    16
)

core_pts = core_fill(
    r_inner_core,
    r_outer_core,
    N_r=4,
    N_theta=11,
    N_phi=11
)

interior_pts = np.array(
    outer_pts
    + inner_pts
    + plane_x_pts
    + plane_y_pts
    + plane_z_pts
    + core_pts,
    dtype=np.float64
)

# ======================================================
# Boundary points
# ======================================================
boundary_pts = []

boundary_pts += spherical_layer_points_positive_octant(
    R_inner,
    8,
    8
)

boundary_pts += spherical_layer_points_positive_octant(
    R_outer,
    16,
    16
)

grid = np.linspace(0.0, R_outer, 16)

for y in grid:
    for z in grid:

        r = np.sqrt(y**2 + z**2)

        if R_inner <= r <= R_outer:

            boundary_pts.append([0.0, y, z])

for x in grid:
    for z in grid:

        r = np.sqrt(x**2 + z**2)

        if R_inner <= r <= R_outer:

            boundary_pts.append([x, 0.0, z])

for x in grid:
    for y in grid:

        r = np.sqrt(x**2 + y**2)

        if R_inner <= r <= R_outer:

            boundary_pts.append([x, y, 0.0])

boundary_pts = np.array(
    boundary_pts,
    dtype=np.float64
)

# ======================================================
# Remove near duplicates
# ======================================================
interior_pts = remove_near_duplicates(
    interior_pts,
    tol=1e-12
)

boundary_pts = remove_near_duplicates(
    boundary_pts,
    tol=1e-12
)

print("Interior points =", interior_pts.shape[0])

print("Boundary points =", boundary_pts.shape[0])

# ======================================================
# Save datasets
# ======================================================
os.makedirs(
    "data",
    exist_ok=True
)
np.save(
    "data/shell_interior_points.npy",
    interior_pts
)

np.save(
    "data/shell_boundary_points.npy",
    boundary_pts
)
# ======================================================
# Visualization
# ======================================================
fig = plt.figure(figsize=(5, 5))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# ------------------------------------------------------
# Scatter plots
# ------------------------------------------------------
ax.scatter(
    interior_pts[:, 0],
    interior_pts[:, 1],
    interior_pts[:, 2],
    s=1,
    color="blue",
    label="Interior"
)

ax.scatter(
    boundary_pts[:, 0],
    boundary_pts[:, 1],
    boundary_pts[:, 2],
    s=1,
    color="red",
    label="Boundary"
)

# ------------------------------------------------------
# Axis labels
# ------------------------------------------------------
ax.set_xlabel(
    r"$x$",
    fontsize=12
)

ax.set_ylabel(
    r"$y$",
    fontsize=12
)

ax.set_zlabel(
    r"$z$",
    fontsize=12,
    labelpad=0.5
)

# ------------------------------------------------------
# Uniform cube scaling
# ------------------------------------------------------
xmin = min(
    interior_pts[:, 0].min(),
    boundary_pts[:, 0].min()
)

xmax = max(
    interior_pts[:, 0].max(),
    boundary_pts[:, 0].max()
)

ymin = min(
    interior_pts[:, 1].min(),
    boundary_pts[:, 1].min()
)

ymax = max(
    interior_pts[:, 1].max(),
    boundary_pts[:, 1].max()
)

zmin = min(
    interior_pts[:, 2].min(),
    boundary_pts[:, 2].min()
)

zmax = max(
    interior_pts[:, 2].max(),
    boundary_pts[:, 2].max()
)

min_val = min(xmin, ymin, zmin)

max_val = max(xmax, ymax, zmax)

ax.set_xlim(min_val, max_val)

ax.set_ylim(min_val, max_val)

ax.set_zlim(min_val, max_val)

# ------------------------------------------------------
# Equal aspect ratio
# ------------------------------------------------------
ax.set_box_aspect([1, 1, 1])

# ------------------------------------------------------
# Legend
# ------------------------------------------------------
ax.legend(
    loc="upper left",
    frameon=False,
    fontsize=10,
    markerscale=7
)

plt.tight_layout()

# ======================================================
# Save figure
# ======================================================
plt.savefig(
    "Figure7_a.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




