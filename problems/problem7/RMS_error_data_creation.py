#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree

# =====================================================
# Shell parameters
# =====================================================
R_inner = 0.5

R_outer = 1.0

# =====================================================
# Create data folder
# =====================================================
os.makedirs(
    "data",
    exist_ok=True
)

# =====================================================
# Near-duplicate remover
# =====================================================
def remove_near_duplicates(
    points,
    tol=1e-12
):

    tree = cKDTree(points)

    keep = np.ones(
        len(points),
        dtype=bool
    )

    for i, p in enumerate(points):

        if not keep[i]:
            continue

        idx = tree.query_ball_point(
            p,
            tol
        )

        idx.remove(i)

        keep[idx] = False

    return points[keep]

# =====================================================
# SAFE spherical sampling
# =====================================================
def spherical_layer_points_positive_octant(
    r,
    N_theta,
    N_phi,
    pole_tol=1e-10
):

    pts = []

    theta = (
        np.arange(N_theta) + 0.5
    ) * (0.5 * np.pi / N_theta)

    phi = (
        np.arange(N_phi) + 0.5
    ) * (0.5 * np.pi / N_phi)

    for ph in phi:

        sin_ph = np.sin(ph)

        if sin_ph < pole_tol:

            pts.append([
                0.0,
                0.0,
                r * np.cos(ph)
            ])

            continue

        for th in theta:

            x = r * sin_ph * np.cos(th)

            y = r * sin_ph * np.sin(th)

            z = r * np.cos(ph)

            pts.append([x, y, z])

    return pts

# =====================================================
# Radial layers
# =====================================================
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

# =====================================================
# Planar layers
# =====================================================
def planar_boundary_layers(
    axis,
    n_layers,
    base_spacing,
    growth,
    N_plane
):

    pts = []

    d = base_spacing

    grid = np.linspace(
        0.0,
        R_outer,
        N_plane
    )

    for _ in range(n_layers):

        if axis == "x":

            for y in grid:
                for z in grid:

                    r = np.sqrt(
                        d**2 + y**2 + z**2
                    )

                    if R_inner <= r <= R_outer:

                        pts.append([d, y, z])

        elif axis == "y":

            for x in grid:
                for z in grid:

                    r = np.sqrt(
                        x**2 + d**2 + z**2
                    )

                    if R_inner <= r <= R_outer:

                        pts.append([x, d, z])

        elif axis == "z":

            for x in grid:
                for y in grid:

                    r = np.sqrt(
                        x**2 + y**2 + d**2
                    )

                    if R_inner <= r <= R_outer:

                        pts.append([x, y, d])

        d *= growth

    return pts

# =====================================================
# Core fill
# =====================================================
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

# =====================================================
# Configurations
# =====================================================
configs = [

    dict(
        Nt_out=14,
        Np_out=14,
        Nt_in=6,
        Np_in=6,
        N_plane=14,
        Nr=2,
        Nt_core=9,
        Np_core=9
    ),

    dict(
        Nt_out=15,
        Np_out=15,
        Nt_in=7,
        Np_in=7,
        N_plane=15,
        Nr=3,
        Nt_core=10,
        Np_core=10
    ),

    dict(
        Nt_out=16,
        Np_out=16,
        Nt_in=8,
        Np_in=8,
        N_plane=16,
        Nr=4,
        Nt_core=11,
        Np_core=11
    ),

    dict(
        Nt_out=17,
        Np_out=17,
        Nt_in=9,
        Np_in=9,
        N_plane=17,
        Nr=5,
        Nt_core=12,
        Np_core=12
    ),

    dict(
        Nt_out=18,
        Np_out=18,
        Nt_in=10,
        Np_in=10,
        N_plane=18,
        Nr=6,
        Nt_core=13,
        Np_core=13
    ),
]

# =====================================================
# Generate datasets
# =====================================================
for k, cfg in enumerate(configs, start=1):

    outer_pts, r_outer_core = radial_layers_outer(
        R_inner,
        R_outer,
        2,
        0.01,
        1.3,
        cfg["Nt_out"],
        cfg["Np_out"]
    )

    inner_pts, r_inner_core = radial_layers_inner(
        R_inner,
        R_outer,
        2,
        0.01,
        1.3,
        cfg["Nt_in"],
        cfg["Np_in"]
    )

    plane_x = planar_boundary_layers(
        "x",
        2,
        0.01,
        1.3,
        cfg["N_plane"]
    )

    plane_y = planar_boundary_layers(
        "y",
        2,
        0.01,
        1.3,
        cfg["N_plane"]
    )

    plane_z = planar_boundary_layers(
        "z",
        2,
        0.01,
        1.3,
        cfg["N_plane"]
    )

    core = core_fill(
        r_inner_core,
        r_outer_core,
        cfg["Nr"],
        cfg["Nt_core"],
        cfg["Np_core"]
    )

    interior_pts = np.array(
        outer_pts
        + inner_pts
        + plane_x
        + plane_y
        + plane_z
        + core,
        dtype=np.float64
    )

    boundary_pts = []

    boundary_pts += spherical_layer_points_positive_octant(
        R_inner,
        cfg["Nt_in"],
        cfg["Np_in"]
    )

    boundary_pts += spherical_layer_points_positive_octant(
        R_outer,
        cfg["Nt_out"],
        cfg["Np_out"]
    )

    grid = np.linspace(
        0.0,
        R_outer,
        cfg["N_plane"]
    )

    for y in grid:
        for z in grid:

            if (
                R_inner
                <= np.sqrt(y**2 + z**2)
                <= R_outer
            ):

                boundary_pts.append([
                    0.0,
                    y,
                    z
                ])

    for x in grid:
        for z in grid:

            if (
                R_inner
                <= np.sqrt(x**2 + z**2)
                <= R_outer
            ):

                boundary_pts.append([
                    x,
                    0.0,
                    z
                ])

    for x in grid:
        for y in grid:

            if (
                R_inner
                <= np.sqrt(x**2 + y**2)
                <= R_outer
            ):

                boundary_pts.append([
                    x,
                    y,
                    0.0
                ])

    boundary_pts = np.array(
        boundary_pts,
        dtype=np.float64
    )

    # =================================================
    # Remove near duplicates
    # =================================================
    interior_pts = remove_near_duplicates(
        interior_pts
    )

    boundary_pts = remove_near_duplicates(
        boundary_pts
    )

    # =================================================
    # Save datasets
    # =================================================
    np.save(
        f"data/shell_i{k}.npy",
        interior_pts
    )

    np.save(
        f"data/shell_b{k}.npy",
        boundary_pts
    )

    total_pts = (
        interior_pts.shape[0]
        + boundary_pts.shape[0]
    )

    print(
        f"Set {k}: "
        f"Interior = {interior_pts.shape[0]}, "
        f"Boundary = {boundary_pts.shape[0]}, "
        f"Total = {total_pts}"
    )

print(
    "\nAll 5 collocation datasets saved."
)


# In[ ]:




