from __future__ import annotations

import numpy as np

# Optional numba support
try:
    from numba import njit, prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    prange = range  # fallback


def one_step_crawl(
    lines: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    point_of_attraction: np.ndarray,
    line_length: float,
    *,
    method: int = 1,
) -> np.ndarray:
    """
    Move each line segment center to a selected point among points within its mask.

    Parameters
    ----------
    lines : (M,4) array
        Each row: [x1, y1, x2, y2]
    points : (N,3) array
        Point cloud XYZ
    mask : (N,M) bool array
        mask[n,i] True if point n belongs to line i's neighborhood
    point_of_attraction : (M,3) array
        Updated in-place with [x,y,H] where H = max(z)-min(z) for that line's points
    line_length : float
        New segment length
    method : int
        1=max(z), 2=min(z), 3=first point with z>=90th percentile (fallback to max)

    Returns
    -------
    endpoints : (M,5) float array
        [new_x1, new_y1, new_x2, new_y2, H] for moved lines.
        Rows remain 0 if a line had no points.
    """
    M = lines.shape[0]
    endpoints = np.zeros((M, 5), dtype=np.float64)

    for i in prange(M):
        old_center_x = 0.5 * (lines[i, 0] + lines[i, 2])
        old_center_y = 0.5 * (lines[i, 1] + lines[i, 3])

        pts_i = points[mask[:, i]]
        if pts_i.shape[0] == 0:
            continue

        col_z = pts_i[:, 2]
        H = float(np.max(col_z) - np.min(col_z))

        if method == 1:
            idx = int(np.argmax(col_z))
        elif method == 2:
            idx = int(np.argmin(col_z))
        elif method == 3:
            target = np.percentile(col_z, 90)
            idxs = np.where(col_z >= target)[0]
            idx = int(idxs[0]) if idxs.size > 0 else int(np.argmax(col_z))
        else:
            raise ValueError("method must be 1, 2, or 3")

        new_center_x = float(pts_i[idx, 0])
        new_center_y = float(pts_i[idx, 1])

        # Update point_of_attraction (always, if pts exist)
        point_of_attraction[i, 0] = new_center_x
        point_of_attraction[i, 1] = new_center_y
        point_of_attraction[i, 2] = H

        # Build new segment endpoints in direction from old to new center
        dx_c = new_center_x - old_center_x
        dy_c = new_center_y - old_center_y
        if dx_c == 0.0 and dy_c == 0.0:
            continue

        theta = np.arctan2(dy_c, dx_c)
        dx = 0.5 * line_length * np.cos(theta)
        dy = 0.5 * line_length * np.sin(theta)

        endpoints[i, 0] = new_center_x - dx
        endpoints[i, 1] = new_center_y - dy
        endpoints[i, 2] = new_center_x + dx
        endpoints[i, 3] = new_center_y + dy
        endpoints[i, 4] = H

    return endpoints
