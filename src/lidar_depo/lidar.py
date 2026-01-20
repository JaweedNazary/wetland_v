from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from shapely.geometry.base import BaseGeometry

from .crs import proj_to_3857, gcs_to_proj
from .pdal_pipeline import build_pdal_pipeline
from .usgs_3dep import ThreeDEPIndex


def _require_lidar_deps():
    try:
        import pdal  # noqa: F401
        import laspy  # noqa: F401
    except Exception as e:
        raise ImportError(
            "PDAL + laspy required. Install with: pip install infiltech-geo[lidar]"
        ) from e


@dataclass(frozen=True)
class LidarPoints:
    ground: np.ndarray
    vegetation: np.ndarray
    water: np.ndarray
    all_points: np.ndarray
    most_recent_year: Optional[int]


def get_lidar_points_around_geometry(
    geom_3857: BaseGeometry,
    index: ThreeDEPIndex,
    *,
    buffer_distance: float = 3,
    res: float = 2,
    out_name: str = "sample_line",
    prefer_year: Optional[int] = None,
) -> LidarPoints:
    """
    Buffers geometry (in EPSG:3857), finds containing 3DEP datasets, runs PDAL,
    returns classified points via laspy.

    If prefer_year is given, only that year is accepted (when parsable).
    """
    _require_lidar_deps()
    import pdal
    import laspy

    geom_buff = geom_3857.buffer(buffer_distance, cap_style=3)

    # Your proj_to_3857/gcs_to_proj pattern is a bit mixed; keep it consistent:
    # Here geom is 3857 already; create WKT directly.
    aoi_wkt = geom_buff.wkt

    intersecting = []
    most_recent = None

    for i, poly_3857 in enumerate(index.geometries_3857):
        if not poly_3857.contains(geom_buff):
            continue

        name = str(index.names.iloc[i])
        nums = re.findall(r"\d+", name)
        yr = None
        if nums:
            # try last 2 digits or 4 digits; handle both
            val = nums[-1]
            yr = int(val) if len(val) == 4 else int(val)

        if prefer_year is not None and yr is not None and yr != prefer_year:
            continue

        intersecting.append(name)
        if yr is not None:
            most_recent = yr if (most_recent is None or yr > most_recent) else most_recent

    if not intersecting:
        raise ValueError("No 3DEP dataset polygon contains the AOI (or year filter excluded all).")

    pc_pipeline_dict = build_pdal_pipeline(
        aoi_wkt,
        intersecting,
        pc_resolution=res,
        filter_noise=False,
        reclassify=False,
        save_pointcloud=True,
        out_crs=3857,
        pc_out_name=out_name,
        pc_out_type="las",
        debug=True,
    )

    pipe = pdal.Pipeline(json.dumps(pc_pipeline_dict))
    pipe.execute()  # single output file

    las = laspy.read(f"{out_name}.las")
    classes = np.unique(las.classification)

    pts = las.points
    ground = pts[las.classification == 2] if 2 in classes else np.array([])
    veg = pts[las.classification == 1] if 1 in classes else np.array([])
    water = pts[las.classification == 9] if 9 in classes else np.array([])

    return LidarPoints(ground=ground, vegetation=veg, water=water, all_points=pts, most_recent_year=most_recent)
