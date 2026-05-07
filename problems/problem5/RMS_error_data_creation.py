#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import (
    Point,
    Polygon,
    MultiPolygon,
    LineString
)
import numpy as np
import os

# ==========================================================
# 1️⃣ Load geometry (Haryana)
# ==========================================================
url = (
    "https://geodata.ucdavis.edu/gadm/"
    "gadm4.1/json/gadm41_IND_1.json"
)

gdf = gpd.read_file(url)

geom = gdf[
    gdf["NAME_1"] == "Haryana"
].geometry.values[0]

if isinstance(geom, MultiPolygon):

    geom = max(
        geom.geoms,
        key=lambda g: g.area
    )

# ==========================================================
# 2️⃣ Rescale geometry to [-1,1] × [-1,1]
# ==========================================================
x0, y0 = geom.exterior.xy

xmin = min(x0)
xmax = max(x0)

ymin = min(y0)
ymax = max(y0)

def rescale(pt):

    return (
        2.0 * (pt[0] - xmin) / (xmax - xmin) - 1.0,
        2.0 * (pt[1] - ymin) / (ymax - ymin) - 1.0,
    )

domain = Polygon([
    rescale(p)
    for p in geom.exterior.coords
])

# ==========================================================
# 3️⃣ Boundary-layer generator
# ==========================================================
def boundary_layer_points(
    domain,
    n_layers,
    base_spacing,
    growth,
    points_per_layer
):

    pts = []

    d = base_spacing

    inner_polygon = domain

    for _ in range(n_layers):

        candidate = inner_polygon.buffer(-d)

        if candidate.is_empty:

            break

        if isinstance(candidate, MultiPolygon):

            candidate = max(
                candidate.geoms,
                key=lambda g: g.area
            )

        ring = LineString(
            candidate.exterior.coords
        )

        L = ring.length

        s_vals = np.linspace(
            0,
            L,
            points_per_layer,
            endpoint=False
        )

        for s in s_vals:

            p = ring.interpolate(s)

            pts.append([p.x, p.y])

        inner_polygon = candidate

        d *= growth

    return np.array(pts), inner_polygon

# ==========================================================
# 4️⃣ Interior fill
# ==========================================================
def interior_fill(domain, N):

    xmin, ymin, xmax, ymax = domain.bounds

    xs = np.linspace(xmin, xmax, N)

    ys = np.linspace(ymin, ymax, N)

    pts = []

    for x in xs:

        for y in ys:

            if domain.contains(Point(x, y)):

                pts.append([x, y])

    return np.array(pts)

# ==========================================================
# 5️⃣ Configuration for 5 datasets
# ==========================================================
configs = [
    dict(Nb=500,  n_layers=1, pts_layer=200, Nint=15),
    dict(Nb=700,  n_layers=2, pts_layer=300, Nint=20),
    dict(Nb=900,  n_layers=2, pts_layer=400, Nint=25),
    dict(Nb=1100, n_layers=2, pts_layer=500, Nint=30),
    dict(Nb=1300, n_layers=2, pts_layer=600, Nint=35),
]

# ==========================================================
# 6️⃣ Create data folder
# ==========================================================
os.makedirs(
    "data",
    exist_ok=True
)

# ==========================================================
# 7️⃣ Generate & save all datasets
# ==========================================================
for k, cfg in enumerate(configs, start=1):

    # ------------------------------------------------------
    # Boundary points
    # ------------------------------------------------------
    boundary_line = LineString(
        domain.exterior.coords
    )

    s = np.linspace(
        0,
        boundary_line.length,
        cfg["Nb"],
        endpoint=False
    )

    boundary_pts = np.array([
        [
            boundary_line.interpolate(si).x,
            boundary_line.interpolate(si).y
        ]
        for si in s
    ])

    # ------------------------------------------------------
    # Interior points
    # ------------------------------------------------------
    bl_pts, core = boundary_layer_points(
        domain,
        n_layers=cfg["n_layers"],
        base_spacing=0.01,
        growth=1.2,
        points_per_layer=cfg["pts_layer"]
    )

    core_pts = interior_fill(
        core,
        cfg["Nint"]
    )

    interior_pts = np.vstack([
        bl_pts,
        core_pts
    ])

    # ------------------------------------------------------
    # Save as .npy files
    # ------------------------------------------------------
    np.save(
        f"data/int_points_{k}.npy",
        interior_pts
    )

    np.save(
        f"data/bnd_points_{k}.npy",
        boundary_pts
    )

    # ------------------------------------------------------
    # Print info
    # ------------------------------------------------------
    print(f"Set {k}:")

    print(
        f"  Interior points = {interior_pts.shape[0]}"
    )

    print(
        f"  Boundary points = {boundary_pts.shape[0]}"
    )

    print("-" * 40)


# In[ ]:




