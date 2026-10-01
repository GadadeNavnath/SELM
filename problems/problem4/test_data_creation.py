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

# =====================================================
# Load geometry
# =====================================================

url = (
    "https://geodata.ucdavis.edu/gadm/"
    "gadm4.1/json/gadm41_IND_1.json"
)

gdf = gpd.read_file(url)

geom = gdf[
    gdf["NAME_1"] == "Chhattisgarh"
].geometry.values[0]

if isinstance(geom, MultiPolygon):

    geom = max(
        geom.geoms,
        key=lambda g: g.area
    )

# =====================================================
# Rescale geometry to [0,10] × [0,10]
# =====================================================

x0, y0 = geom.exterior.xy

xmin = min(x0)
xmax = max(x0)

ymin = min(y0)
ymax = max(y0)

def rescale(pt):

    return (
        10 * (pt[0] - xmin) / (xmax - xmin),
        10 * (pt[1] - ymin) / (ymax - ymin)
    )

domain = Polygon([
    rescale(p)
    for p in geom.exterior.coords
])

# =====================================================
# Uniform boundary points
# =====================================================

Nb = 2000

boundary_line = LineString(
    domain.exterior.coords
)

s = np.linspace(
    0,
    boundary_line.length,
    Nb,
    endpoint=False
)

boundary_pts = np.array([
    [
        boundary_line.interpolate(si).x,
        boundary_line.interpolate(si).y
    ]
    for si in s
])

# =====================================================
# Boundary-layer points
# =====================================================

def boundary_layer_points(
    domain,
    n_layers=6,
    base_spacing=0.001,
    growth=1.5,
    points_per_layer=500
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

# =====================================================
# Interior fill points
# =====================================================

def interior_fill(domain, N=50):

    xmin, ymin, xmax, ymax = domain.bounds

    xs = np.linspace(xmin, xmax, N)

    ys = np.linspace(ymin, ymax, N)

    pts = []

    for x in xs:

        for y in ys:

            if domain.contains(Point(x, y)):

                pts.append([x, y])

    return np.array(pts)

# =====================================================
# Generate interior points
# =====================================================

boundary_layer_pts, core_domain = (
    boundary_layer_points(
        domain,
        n_layers=5,
        base_spacing=0.01,
        growth=1.2,
        points_per_layer=1000
    )
)

interior_fill_pts = interior_fill(
    core_domain,
    N=250
)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])

# =====================================================
# Save arrays
# =====================================================
os.makedirs(
    "data",
    exist_ok=True
)
np.save(
    "data/chhattisgarh_interior_points_test.npy",
    interior_pts
)

np.save(
    "data/chhattisgarh_boundary_points_test.npy",
    boundary_pts
)

# =====================================================
# Information
# =====================================================

print("Boundary points:", boundary_pts.shape[0])

print(
    "Boundary-layer points:",
    boundary_layer_pts.shape[0]
)

print(
    "Interior core points:",
    interior_fill_pts.shape[0]
)

print(
    "Total interior points:",
    interior_pts.shape[0]
)


# In[ ]:




