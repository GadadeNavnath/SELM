#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


# ============================================================
# 2D Bratu problem using P2 FEM
# ============================================================

lam = 6.8081244226

# Mesh divisions in each direction
# Start with n = 50 or 100
n = 200

tol = 1e-10
maxit = 30


# ============================================================
# P1 vertex mesh
# ============================================================

def generate_p1_mesh(n):

    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    X, Y = np.meshgrid(x, y, indexing="ij")

    nodes = np.column_stack([
        X.ravel(),
        Y.ravel()
    ])

    elements = []

    def node(i, j):
        return i * (n + 1) + j

    for i in range(n):
        for j in range(n):

            n1 = node(i, j)
            n2 = node(i + 1, j)
            n3 = node(i + 1, j + 1)
            n4 = node(i, j + 1)

            # Two triangles
            elements.append([n1, n2, n3])
            elements.append([n1, n3, n4])

    return nodes, np.array(elements, dtype=int)


# ============================================================
# Construct P2 mesh
#
# Each P1 triangle gets three mid-edge nodes.
# ============================================================

def generate_p2_mesh(n):

    vertices, triangles = generate_p1_mesh(n)

    p2_nodes = vertices.tolist()

    edge_to_mid = {}

    p2_elements = []

    def get_midpoint(a, b):

        edge = tuple(sorted((a, b)))

        if edge not in edge_to_mid:

            xa, ya = vertices[a]
            xb, yb = vertices[b]

            xm = 0.5 * (xa + xb)
            ym = 0.5 * (ya + yb)

            idx = len(p2_nodes)

            p2_nodes.append([xm, ym])
            edge_to_mid[edge] = idx

        return edge_to_mid[edge]

    for tri in triangles:

        a, b, c = tri

        mab = get_midpoint(a, b)
        mbc = get_midpoint(b, c)
        mca = get_midpoint(c, a)

        # P2 local ordering:
        # vertex 1, vertex 2, vertex 3,
        # edge 12, edge 23, edge 31

        p2_elements.append([
            a, b, c,
            mab, mbc, mca
        ])

    return (
        np.array(p2_nodes, dtype=float),
        np.array(p2_elements, dtype=int)
    )


# ============================================================
# P2 shape functions and derivatives
#
# Reference triangle:
#
# r >= 0, s >= 0, r+s <= 1
# ============================================================

def p2_shape(r, s):

    l1 = 1.0 - r - s
    l2 = r
    l3 = s

    N = np.array([
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1
    ])

    dNdr = np.array([
        -(4.0 * l1 - 1.0),
        4.0 * l2 - 1.0,
        0.0,
        4.0 * (l1 - l2),
        4.0 * l3,
        -4.0 * l3
    ])

    dNds = np.array([
        -(4.0 * l1 - 1.0),
        0.0,
        4.0 * l3 - 1.0,
        -4.0 * l2,
        4.0 * l2,
        4.0 * (l1 - l3)
    ])

    return N, dNdr, dNds


# ============================================================
# 7-point quadrature rule on reference triangle
# Degree-5 accurate
# ============================================================

def triangle_quadrature():

    points = [
        (1/3, 1/3, 0.225000000000000),

        (0.059715871789770, 0.470142064105115,
         0.132394152788506),

        (0.470142064105115, 0.059715871789770,
         0.132394152788506),

        (0.470142064105115, 0.470142064105115,
         0.132394152788506),

        (0.797426985353087, 0.101286507323456,
         0.125939180544827),

        (0.101286507323456, 0.797426985353087,
         0.125939180544827),

        (0.101286507323456, 0.101286507323456,
         0.125939180544827)
    ]

    return points


# ============================================================
# Assemble residual and Jacobian
#
# R(u) = K u + lam * integral(exp(u) N)
#
# Because the PDE is
#
#     u_xx + u_yy + lam exp(u) = 0
#
# multiplying by test function and integrating:
#
#     - integral(grad u . grad v)
#     + lam integral(exp(u)v) = 0
#
# We multiply the complete equation by -1:
#
#     integral(grad u . grad v)
#     - lam integral(exp(u)v) = 0
#
# ============================================================

def assemble_system(nodes, elements, u):

    ndof = len(nodes)

    R = np.zeros(ndof)

    J = lil_matrix((ndof, ndof))

    quad = triangle_quadrature()

    for elem in elements:

        coords = nodes[elem[:3]]

        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        # Jacobian of affine P1 geometry
        B = np.array([
            [x2 - x1, x3 - x1],
            [y2 - y1, y3 - y1]
        ])

        detB = np.linalg.det(B)

        if detB <= 0:
            raise RuntimeError("Element orientation error.")

        invB = np.linalg.inv(B)

        u_local = u[elem]

        Rloc = np.zeros(6)
        Jloc = np.zeros((6, 6))

        for r, s, w in quad:

            N, dNdr, dNds = p2_shape(r, s)

            # Gradients with respect to physical coordinates
            grad_ref = np.vstack([
                dNdr,
                dNds
            ])

            grad_phys = invB.T @ grad_ref

            # Solution at quadrature point
            uq = N @ u_local

            exp_u = np.exp(uq)

            weight = w * detB

            # Weak residual
            Rloc += (
                grad_phys.T @ grad_phys
            ) @ u_local * weight

            Rloc -= (
                lam * exp_u * N * weight
            )

            # Jacobian
            Jloc += (
                grad_phys.T @ grad_phys
            ) * weight

            Jloc -= (
                lam * exp_u
                * np.outer(N, N)
                * weight
            )

        # Assemble
        for i in range(6):

            I = elem[i]

            R[I] += Rloc[i]

            for j in range(6):

                J[I, elem[j]] += Jloc[i, j]

    return R, csr_matrix(J)


# ============================================================
# Generate P2 mesh
# ============================================================

t_start = time.time()

nodes, elements = generate_p2_mesh(n)

ndof = len(nodes)

print("")
print("==============================================")
print("2D Bratu P2 FEM")
print("==============================================")
print(f"lambda           = {lam:.10f}")
print(f"Mesh divisions   = {n} x {n}")
print(f"Elements         = {len(elements)}")
print(f"Nodes / DOF      = {ndof}")
print("==============================================")


# ============================================================
# Dirichlet boundary nodes
# ============================================================

boundary = np.where(
    (np.isclose(nodes[:, 0], 0.0)) |
    (np.isclose(nodes[:, 0], 1.0)) |
    (np.isclose(nodes[:, 1], 0.0)) |
    (np.isclose(nodes[:, 1], 1.0))
)[0]

boundary = np.unique(boundary)

free = np.setdiff1d(
    np.arange(ndof),
    boundary
)

print(f"Boundary DOF     = {len(boundary)}")


# ============================================================
# Initial guess
# ============================================================

u = np.zeros(ndof)


# ============================================================
# Newton iteration
# ============================================================

for it in range(maxit):

    R, J = assemble_system(
        nodes,
        elements,
        u
    )

    # Apply homogeneous Dirichlet BC
    R_free = R[free]

    J_free = J[free][:, free]

    residual_norm = np.linalg.norm(R_free)

    if it == 0:
        old_u_norm = 1.0

    # Newton step
    delta = spsolve(
        J_free,
        -R_free
    )

    u_new = u.copy()

    u_new[free] += delta

    # Relative change
    relative_change = (
        np.linalg.norm(u_new - u)
        /
        max(np.linalg.norm(u_new), 1.0)
    )

    print(
        f"Newton {it+1:02d}: "
        f"residual = {residual_norm:.6e}, "
        f"relative change = {relative_change:.6e}"
    )

    u = u_new

    if relative_change < tol:

        print(
            f"Converged at iteration {it+1}"
        )

        break


# ============================================================
# Final results
# ============================================================

R, J = assemble_system(
    nodes,
    elements,
    u
)

final_residual = np.linalg.norm(
    R[free]
)

center = np.argmin(
    (nodes[:, 0] - 0.5)**2
    +
    (nodes[:, 1] - 0.5)**2
)

u_center = u[center]

boundary_error = np.max(
    np.abs(u[boundary])
)

total_time = time.time() - t_start


print("")
print("==============================================")
print("FINAL P2 FEM RESULT")
print("==============================================")
print(f"lambda             = {lam:.10f}")
print(f"u(0.5,0.5)         = {u_center:.10f}")
print(f"Maximum u          = {np.max(u):.10f}")
print(f"Interior residual   = {final_residual:.6e}")
print(f"Boundary error      = {boundary_error:.6e}")
print(f"Total time          = {total_time:.5f} s")


# ============================================================
# Comparison with published critical reference
# ============================================================

reference = 1.3916612060

abs_error = abs(
    u_center - reference
)

rel_error = (
    abs_error
    /
    abs(reference)
    * 100.0
)

print("")
print("==============================================")
print("COMPARISON WITH PUBLISHED REFERENCE")
print("==============================================")
print(f"P2 FEM              = {u_center:.10f}")
print(f"Published reference = {reference:.10f}")
print(f"Absolute error      = {abs_error:.6e}")
print(f"Relative error      = {rel_error:.6f} %")


# ============================================================
# #Save FEM solution
# ============================================================

np.savetxt(
    "fem.csv",
    np.column_stack([
        nodes[:, 0],
        nodes[:, 1],
        u
    ]),
    delimiter=",",
    header="x,y,u_P2_FEM",
    comments=""
)

print("")
print("Solution saved to:")
print("fem.csv")


# In[ ]:




