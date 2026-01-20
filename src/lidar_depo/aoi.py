from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape

from .crs import proj_to_3857, gcs_to_proj, CRSLike


@dataclass(frozen=True)
class AOI:
    geom_4326: BaseGeometry
    geom_3857: BaseGeometry

    @property
    def bounds_3857(self):
        return self.geom_3857.bounds


def import_shapefile_to_aoi(path: Union[str, Path]) -> AOI:
    gdf = gpd.read_file(str(path))
    if gdf.empty:
        raise ValueError("Shapefile has no features.")
    orig_crs = gdf.crs
    geom = gdf.loc[gdf.index[0], "geometry"]
    geom_4326, geom_3857 = proj_to_3857(geom, orig_crs)
    return AOI(geom_4326=geom_4326, geom_3857=geom_3857)


def aoi_from_geojson_geometry(geo_json: dict) -> AOI:
    """Takes a GeoJSON feature or geometry dict and returns AOI in 4326 + 3857."""
    geom_dict = geo_json["geometry"] if "geometry" in geo_json else geo_json
    geom_4326 = shape(geom_dict)
    geom_3857 = gcs_to_proj(geom_4326)
    return AOI(geom_4326=geom_4326, geom_3857=geom_3857)
