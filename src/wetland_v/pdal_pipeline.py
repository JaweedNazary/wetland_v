from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def build_pdal_pipeline(
    extent_epsg3857_wkt: str,
    usgs_3dep_dataset_names: List[str],
    pc_resolution: float,
    *,
    filter_noise: bool = False,
    reclassify: bool = False,
    save_pointcloud: bool = True,
    out_crs: int = 3857,
    pc_out_name: str = "filter_test",
    pc_out_type: str = "laz",
    debug: bool = False,
) -> Dict:
    readers = []
    for name in usgs_3dep_dataset_names:
        url = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-public/{name}/ept.json"
        readers.append(
            {
                "type": "readers.ept",
                "filename": url,
                "polygon": extent_epsg3857_wkt,
                "requests": 3,
                "resolution": pc_resolution,
            }
        )

    pipeline = {"pipeline": readers}

    if filter_noise:
        pipeline["pipeline"].append({"type": "filters.range", "limits": "Classification[2:2]"})

    if reclassify:
        pipeline["pipeline"].extend(
            [
                {"type": "filters.assign", "value": "Classification = 0"},
                {"type": "filters.smrf"},
                {"type": "filters.range", "limits": "Classification[2:2]"},
            ]
        )

    pipeline["pipeline"].append({"type": "filters.reprojection", "out_srs": f"EPSG:{out_crs}"})

    if debug:
        pipeline["debug"] = True

    if save_pointcloud:
        if pc_out_type not in {"las", "laz"}:
            raise ValueError("pc_out_type must be 'las' or 'laz'.")
        writer = {"type": "writers.las", "filename": f"{pc_out_name}.{pc_out_type}"}
        if pc_out_type == "laz":
            writer["compression"] = "laszip"
        pipeline["pipeline"].append(writer)

    return pipeline
