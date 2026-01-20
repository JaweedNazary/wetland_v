from __future__ import annotations
import numpy as np


def cross_3d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[0] = u[1]*v[2] - u[2]*v[1]
    result[1] = u[2]*v[0] - u[0]*v[2]
    result[2] = u[0]*v[1] - u[1]*v[0]
    return result


def get_cross_section_mask(lines: np.ndarray, points: np.ndarray, thickness: float = 10):
    M = lines.shape[0]  # number of lines

    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    # Create P0, P1, P2, each shape (M, 3)
    P0 = np.column_stack((x1, y1, np.zeros(M)))
    P1 = np.column_stack((x1, y1, np.full(M, 5.0)))
    P2 = np.column_stack((x2, y2, np.zeros(M)))
    
    # ------------------------------------------------------
    # 2) Compute plane normals N = cross(P1 - P0, P2 - P1)
    # ------------------------------------------------------
    U = P1 - P0  # shape (M,3)
    V = P2 - P1  # shape (M,3)
    
    N = np.empty((M, 3), dtype=np.float64)
    for i in prange(M):
        N[i] = cross_3d(U[i], V[i])
    
    normN = np.sqrt(N[:, 0]**2 + N[:, 1]**2 + N[:, 2]**2)  # shape (M,)
    # Avoid dividing by zero for degenerate lines:
    mask_valid = normN > 1e-12
    for i in range(M):
        if not mask_valid[i]:
            N[i, :] = 0.0
    
    unit_N = np.zeros_like(N)
    for i in range(M):
        if mask_valid[i]:
            unit_N[i] = N[i] / normN[i]
    
    # ------------------------------------------------------
    # 3) Compute distance of each point to each plane.
    #    distance[i,j] = (points[j] - P2[i]) • unit_N[i]
    # ------------------------------------------------------
    plane_point = P2  # shape (M,3)
    num_points = points.shape[0]
    distance = np.empty((num_points, M), dtype=np.float64)
    for i in prange(num_points):
        for j in range(M):
            dx = points[i, 0] - plane_point[j, 0]
            dy = points[i, 1] - plane_point[j, 1]
            dz = points[i, 2] - plane_point[j, 2]
            distance[i, j] = dx*unit_N[j, 0] + dy*unit_N[j, 1] + dz*unit_N[j, 2]
    
    # Keep points within thickness/2:
    plane_mask = np.abs(distance) < (thickness / 2)
    
    # ------------------------------------------------------
    # 4) "Between endpoints" check.
    #    Compute unit vector along the line for each segment.
    # ------------------------------------------------------
    D = P2 - P0  # shape (M,3)
    norm_D = np.sqrt(D[:, 0]**2 + D[:, 1]**2 + D[:, 2]**2).reshape(M, 1)
    valid_D = norm_D[:, 0] > 1e-12  # Boolean array for valid vectors
    unit_D = np.zeros_like(D)
    for i in range(M):
        if valid_D[i]:
            unit_D[i] = D[i] / norm_D[i, 0]
    
    # Compute projections on the line direction
    d0 = np.empty((num_points, M), dtype=np.float64)
    d2 = np.empty((num_points, M), dtype=np.float64)
    for i in prange(num_points):
        for j in range(M):
            # V0 = points[i] - P0[j]
            v0x = points[i, 0] - P0[j, 0]
            v0y = points[i, 1] - P0[j, 1]
            v0z = points[i, 2] - P0[j, 2]
            d0[i, j] = v0x * unit_D[j, 0] + v0y * unit_D[j, 1] + v0z * unit_D[j, 2]
            # V2 = points[i] - P2[j]
            v2x = points[i, 0] - P2[j, 0]
            v2y = points[i, 1] - P2[j, 1]
            v2z = points[i, 2] - P2[j, 2]
            d2[i, j] = v2x * unit_D[j, 0] + v2y * unit_D[j, 1] + v2z * unit_D[j, 2]
    
    between_mask = (d0 >= 0.0) & (d2 <= 0.0)
    
    # ------------------------------------------------------
    # 5) Final mask: a point belongs to cross-section if both conditions hold.
    # ------------------------------------------------------
    cross_section_mask = plane_mask & between_mask
    return cross_section_mask
