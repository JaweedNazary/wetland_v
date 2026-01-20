from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform


CRSLike = Union[str, pyproj.CRS]


def proj_to_3857(poly: BaseGeometry, orig_crs: CRSLike) -> Tuple[BaseGeometry, BaseGeometry]:
    """
    Project a geometry from orig_crs into:
      - EPSG:4326 (WGS84)
      - EPSG:3857 (Web Mercator)

    Returns (geom_4326, geom_3857).
    """
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")

    orig = pyproj.CRS.from_user_input(orig_crs)
    to_4326 = pyproj.Transformer.from_crs(orig, wgs84, always_xy=True).transform
    to_3857 = pyproj.Transformer.from_crs(orig, web_mercator, always_xy=True).transform

    return transform(to_4326, poly), transform(to_3857, poly)


def gcs_to_proj(poly: BaseGeometry) -> BaseGeometry:
    """EPSG:4326 -> EPSG:3857"""
    wgs84 = pyproj.CRS("EPSG:4326")
    web_mercator = pyproj.CRS("EPSG:3857")
    project = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True).transform
    return transform(project, poly)
