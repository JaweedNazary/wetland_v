from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests

from .crs import gcs_to_proj


DEFAULT_3DEP_URL = "https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson"


@dataclass(frozen=True)
class ThreeDEPIndex:
    """
    Container for the USGS 3DEP boundary index.

    Attributes
    ----------
    gdf : GeoDataFrame
        Original polygons in their original CRS (usually EPSG:4326).
    geometries_3857 : GeoSeries
        Same polygons projected to EPSG:3857.
    """
    gdf: gpd.GeoDataFrame
    geometries_3857: gpd.GeoSeries

    @property
    def names(self) -> pd.Series:
        return self.gdf["name"]

    @property
    def urls(self) -> pd.Series:
        return self.gdf["url"]

    @property
    def counts(self) -> pd.Series:
        return self.gdf["count"]

    @property
    def geometries_gcs(self) -> gpd.GeoSeries:
        return self.gdf.geometry

    @property
    def years_from_name_last4(self) -> pd.Series:
        # matches your original: year = df["name"].str[-4:]
        return self.gdf["name"].astype(str).str[-4:]


def load_3dep_index(url: str = DEFAULT_3DEP_URL, *, timeout: int = 60) -> ThreeDEPIndex:
    """
    Download and load the 3DEP dataset polygons (resources.geojson).

    This does NOT write to disk (unlike your notebook).
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    # geopandas can read a GeoJSON string directly
    gdf = gpd.read_file(r.text)

    # Project polygons to EPSG:3857 (needed for EPT polygon queries)
    projected = [gcs_to_proj(geom) for geom in gdf.geometry]
    geoms_3857 = gpd.GeoSeries(projected, crs="EPSG:3857")

    return ThreeDEPIndex(gdf=gdf, geometries_3857=geoms_3857)
