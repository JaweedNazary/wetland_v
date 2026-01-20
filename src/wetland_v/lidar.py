from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry.base import BaseGeometry

from .pdal_pipeline import build_pdal_pipeline
from .usgs_3dep import ThreeDEPIndex


def _require_lidar_deps():
    try:
        import pdal  # noqa: F401
        import laspy  # noqa: F401
    except Exception as e:
        raise ImportError(
            "PDAL + laspy required. Install with:\n"
            "  pip install -e \".[lidar]\"\n"
            "or\n"
            "  pip install \"wetland_v[lidar] @ git+https://github.com/JaweedNazary/wetland_v.git\""
        ) from e


def _parse_year_4digit(name: str) -> Optional[int]:
    m = re.search(r"(19\d{2}|20\d{2})", name)
    return int(m.group(1)) if m else None


@dataclass(frozen=True)
class LidarPoints:
    ground_xyz: np.ndarray
    veg_xyz: np.ndarray
    water_xyz: np.ndarray
    all_xyz: np.ndarray
    most_recent_year: Optional[int]
    las_path: Path


def get_lidar_points_around_geometry_3857(
    geom_3857: BaseGeometry,
    index: ThreeDEPIndex,
    *,
    buffer_distance: float = 3.0,
    res: float = 2.0,
    out_name: str = "sample_line",
    out_dir: str | Path = ".",
    prefer_year: Optional[int] = None,
    debug: bool = False,
) -> LidarPoints:
    """
    EPSG:3857-only workflow.

    - Buffers geom (meters)
    - Finds intersecting 3DEP polygons
    - Runs PDAL EPT->LAS
    - Returns Nx3 arrays (x,y,z)

    prefer_year expects a 4-digit year (e.g., 2019).
    """
    _require_lidar_deps()
    import pdal
    import laspy

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    las_path = out_dir / f"{out_name}.las"

    aoi = geom_3857.buffer(buffer_distance, cap_style=3)
    aoi_wkt = aoi.wkt

    datasets: list[str] = []
    most_recent: Optional[int] = None

    for i, poly in enumerate(index.geometries_3857):
        if not poly.intersects(aoi):
            continue

        name = str(index.names.iloc[i])
        yr = _parse_year_4digit(name)

        if prefer_year is not None:
            if yr is None or yr != prefer_year:
                continue

        datasets.append(name)
        if yr is not None:
            most_recent = yr if (most_recent is None or yr > most_recent) else most_recent

    if debug:
        print(f"AOI bounds (3857): {aoi.bounds}")
        print(f"Intersecting datasets: {len(datasets)}")
        if datasets:
            print("First few:", datasets[:5])
        print("Most recent year:", most_recent)

    if not datasets:
        raise ValueError("No 3DEP polygon intersects the AOI (or year filter excluded all).")

    # IMPORTANT: match your original build_pdal_pipeline signature exactly:
    pipeline_dict = build_pdal_pipeline(
        extent_epsg3857=aoi_wkt,
        usgs_3dep_dataset_names=datasets,
        pc_resolution=res,
        filterNoise=False,
        reclassify=False,
        savePointCloud=True,
        outCRS=3857,
        pc_outName=str(las_path.with_suffix("")),  # base path without extension
        pc_outType="las",
        debug=debug,
    )

    pipe = pdal.Pipeline(json.dumps(pipeline_dict))
    pipe.execute()

    if not las_path.exists():
        raise FileNotFoundError(f"Expected LAS not found: {las_path}")

    las = laspy.read(str(las_path))
    all_xyz = np.column_stack([las.x, las.y, las.z])

    cls = las.classification
    ground_xyz = all_xyz[cls == 2] if np.any(cls == 2) else np.empty((0, 3))
    veg_xyz = all_xyz[cls == 1] if np.any(cls == 1) else np.empty((0, 3))
    water_xyz = all_xyz[cls == 9] if np.any(cls == 9) else np.empty((0, 3))

    if debug:
        print("Total points:", all_xyz.shape[0])
        print("Ground:", ground_xyz.shape[0], "Veg:", veg_xyz.shape[0], "Water:", water_xyz.shape[0])

    return LidarPoints(
        ground_xyz=ground_xyz,
        veg_xyz=veg_xyz,
        water_xyz=water_xyz,
        all_xyz=all_xyz,
        most_recent_year=most_recent,
        las_path=las_path,
    )
