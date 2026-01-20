import re
from typing import List
from shapely.geometry.base import BaseGeometry

from .crs import proj_to_3857, gcs_to_proj
from .usgs_3dep import ThreeDEPIndex


def get_available_years(point_geom: BaseGeometry, index: ThreeDEPIndex, buffer_distance: float = 3) -> List[int]:
    geom_buff = point_geom.buffer(buffer_distance, cap_style=3)
    aoi_gcs, _ = proj_to_3857(geom_buff, "EPSG:3857")
    aoi_3857 = gcs_to_proj(aoi_gcs)

    years: List[int] = []

    for i, poly_3857 in enumerate(index.geometries_3857):
        if not poly_3857.contains(aoi_3857):
            continue

        name = str(index.names.iloc[i])
        nums = re.findall(r"\d+", name)
        if nums:
            years.append(int(nums[-1][-2:]))  # keep your original behavior

    return years
