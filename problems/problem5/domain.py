#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import numpy as np
import os

# ==========================================================
# 1️⃣ Load geometry (Haryana)
# ==========================================================
url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_1.json"

gdf = gpd.read_file(url)

geom = gdf[gdf["NAME_1"] == "Haryana"].geometry.values[0]

if isinstance(geom, MultiPolygon):
    geom = max(geom.geoms, key=lambda g: g.area)

# ==========================================================
# 2️⃣ Rescale geometry to [0,2] × [0,2]
# ==========================================================
x0, y0 = geom.exterior.xy

xmin, xmax = min(x0), max(x0)

ymin, ymax = min(y0), max(y0)

def rescale(pt):

    return (
        2.0 * (pt[0] - xmin) / (xmax - xmin),
        2.0 * (pt[1] - ymin) / (ymax - ymin),
    )

domain = Polygon([
    rescale(p)
    for p in geom.exterior.coords
])

# ==========================================================
# 3️⃣ Uniform boundary points
# ==========================================================
Nb = 700

boundary_line = LineString(domain.exterior.coords)

s_vals = np.linspace(
    0,
    boundary_line.length,
    Nb,
    endpoint=False
)

boundary_pts = np.array([
    [
        boundary_line.interpolate(s).x,
        boundary_line.interpolate(s).y
    ]
    for s in s_vals
])

# ==========================================================
# 4️⃣ Boundary-layer points
# ==========================================================
def boundary_layer_points(domain,
                          n_layers=2,
                          base_spacing=0.01,
                          growth=1.2,
                          points_per_layer=300):

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

        ring = LineString(candidate.exterior.coords)

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
# 5️⃣ Interior fill
# ==========================================================
def interior_fill(domain, N=20):

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
# 6️⃣ Generate interior points
# ==========================================================
boundary_layer_pts, core_domain = boundary_layer_points(
    domain,
    n_layers=2,
    base_spacing=0.01,
    growth=1.2,
    points_per_layer=300
)

interior_fill_pts = interior_fill(
    core_domain,
    N=20
)

interior_pts = np.vstack([
    boundary_layer_pts,
    interior_fill_pts
])

# ==========================================================
# 7️⃣ Map from [0,2]^2 → [-1,1]^2
# ==========================================================
def map_to_minus1_1(pts):

    return pts - 1.0

boundary_pts = map_to_minus1_1(boundary_pts)

interior_pts = map_to_minus1_1(interior_pts)

# ==========================================================
# 8️⃣ Save arrays
# ==========================================================
os.makedirs(
    "data",
    exist_ok=True
)
np.save(
    "data/haryana_interior_points.npy",
    interior_pts
)

np.save(
    "data/haryana_boundary_points.npy",
    boundary_pts
)

# ==========================================================
# 9️⃣ Information
# ==========================================================
print("Boundary points shape        :", boundary_pts.shape[0])

print("Boundary-layer points shape  :", boundary_layer_pts.shape[0])

print("Interior core points shape   :", interior_fill_pts.shape[0])

print("Total interior points shape  :", interior_pts.shape[0])


# ==========================================================
# 🔟 Plot
# ==========================================================
plt.figure(figsize=(5, 5))

# ----------------------------------------------------------
# Interior points
# ----------------------------------------------------------
plt.scatter(
    interior_pts[:, 0],
    interior_pts[:, 1],
    s=1,
    c="blue",
    label="Interior",
    zorder=1
)

# ----------------------------------------------------------
# Boundary points
# ----------------------------------------------------------
plt.scatter(
    boundary_pts[:, 0],
    boundary_pts[:, 1],
    s=1,
    c="red",
    label="Boundary",
    zorder=3
)

# ----------------------------------------------------------
# Boundary curve
# ----------------------------------------------------------
boundary_closed = np.vstack([
    boundary_pts,
    boundary_pts[0]
])

plt.plot(
    boundary_closed[:, 0],
    boundary_closed[:, 1],
    linestyle="-",
    linewidth=0.8,
    color="black",
    alpha=0.6,
    zorder=2
)

# ----------------------------------------------------------
# Figure formatting
# ----------------------------------------------------------
plt.xlabel("x")

plt.ylabel("y")

plt.xlim(-1, 1)

plt.ylim(-1, 1)

plt.legend(markerscale=10)

plt.gca().set_aspect(
    "equal",
    adjustable="box"
)

plt.tight_layout()

# ----------------------------------------------------------
# Save figure
# ----------------------------------------------------------
plt.savefig(
    "Figure5_a.pdf",
    bbox_inches="tight"
)

plt.show()


# In[ ]:




