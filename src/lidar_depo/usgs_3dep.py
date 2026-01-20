from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
import requests

from .crs import gcs_to_proj


@dataclass(frozen=True)
class ThreeDEPIndex:
    gdf: gpd.GeoDataFrame
    geometries_4326: gpd.GeoSeries
    geometries_3857: gpd.GeoSeries

    @property
    def names(self):
        return self.gdf["name"]

    @property
    def urls(self):
        return self.gdf["url"]

    @property
    def counts(self):
        return self.gdf["count"]


def load_3dep_index(url: str) -> ThreeDEPIndex:
    """
    Downloads the 3DEP resources GeoJSON and returns a GeoDataFrame + projected geometry series.
    No temp files written.
    """
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    # geopandas can read GeoJSON from text via a temporary in-memory approach:
    gdf = gpd.read_file(r.text)

    projected = [gcs_to_proj(geom) for geom in gdf.geometry]
    geom_3857 = gpd.GeoSeries(projected, crs="EPSG:3857")
    geom_4326 = gdf.geometry

    return ThreeDEPIndex(gdf=gdf, geometries_4326=geom_4326, geometries_3857=geom_3857)
