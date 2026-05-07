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

Nb = 500

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
        n_layers=2,
        base_spacing=0.01,
        growth=1.5,
        points_per_layer=300
    )
)

interior_fill_pts = interior_fill(
    core_domain,
    N=20
)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])

# =====================================================
# Save arrays
# =====================================================

np.save(
    "chhattisgarh_interior_points.npy",
    interior_pts
)

np.save(
    "chhattisgarh_boundary_points.npy",
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

# =====================================================
# Plot points
# =====================================================

plt.figure(figsize=(5, 5))

# Interior points
plt.scatter(
    interior_pts[:, 0],
    interior_pts[:, 1],
    s=1,
    c="blue",
    label="Interior",
    zorder=1
)

# Boundary points
plt.scatter(
    boundary_pts[:, 0],
    boundary_pts[:, 1],
    s=1,
    c="red",
    label="Boundary",
    zorder=3
)

# Closed boundary curve
boundary_closed = np.vstack([
    boundary_pts,
    boundary_pts[0]
])

plt.plot(
    boundary_closed[:, 0],
    boundary_closed[:, 1],
    linestyle='-',
    linewidth=0.8,
    color='black',
    alpha=0.6,
    zorder=2
)

plt.xlabel("x")

plt.ylabel("y")

plt.xlim(0, 10)

plt.ylim(0, 10)

plt.legend(markerscale=10)

plt.gca().set_aspect(
    "equal",
    adjustable="box"
)

plt.tight_layout()

plt.savefig(
    "Figure4_a.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




