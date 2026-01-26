from __future__ import annotations
from typing import Optional, Literal, List


import plotly.graph_objects as go


from shapely.geometry import Polygon, MultiPolygon, box

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.io import MemoryFile

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from pystac_client import Client
import planetary_computer as pc


#################################################
##  LAND USE LAND COVER from SENTINEL 1 & 2    ##
#################################################


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass
class LULCRequest:
    dataset: Literal["worldcover"] = "worldcover"
    year: int = 2020            # 2020 or 2021 for ESA WorldCover on PC :contentReference[oaicite:1]{index=1}
    out_tif: str = "lulc_clip.tif"


def _to_wgs84(aoi_geom, aoi_crs_epsg: int):
    if aoi_crs_epsg == 4326:
        return aoi_geom
    tfm = Transformer.from_crs(f"EPSG:{aoi_crs_epsg}", "EPSG:4326", always_xy=True)
    return shp_transform(lambda x, y, z=None: tfm.transform(x, y), aoi_geom)


def download_LULC(aoi_geom, aoi_crs_epsg: int, req: LULCRequest) -> str:
    """
    Downloads ESA WorldCover (2020 or 2021) tiles intersecting AOI,
    mosaics them if needed, then clips to AOI and writes GeoTIFF.
    """
    if req.year not in (2020, 2021):
        raise ValueError("ESA WorldCover on Planetary Computer supports year=2020 or 2021.")

    # AOI must be WGS84 for STAC intersects
    aoi_wgs84 = _to_wgs84(aoi_geom, aoi_crs_epsg)
    aoi_geojson = mapping(aoi_wgs84)

    # Query by datetime (this is the key fix) :contentReference[oaicite:2]{index=2}
    start = f"{req.year}-01-01"
    end = f"{req.year+1}-01-01"
    time_range = f"{start}/{end}"

    catalog = Client.open(STAC_URL, modifier=pc.sign_inplace)

    search = catalog.search(
        collections=["esa-worldcover"],
        intersects=aoi_geojson,
        datetime=time_range,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError(f"No WorldCover items found for AOI in {time_range}.")

    # Open the "map" asset from each tile (COG)
    srcs: List[rasterio.io.DatasetReader] = []
    try:
        for it in items:
            href = it.assets["map"].href
            srcs.append(rasterio.open(href))

        # Mosaic
        mosaic, out_transform = merge(srcs)   # (bands, H, W)
        mosaic = mosaic.astype(srcs[0].dtypes[0])
        meta = srcs[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "count": mosaic.shape[0],
            }
        )

        # Clip (mask) requires AOI in raster CRS
        raster_crs = srcs[0].crs
        tfm = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        aoi_in_raster = shp_transform(lambda x, y, z=None: tfm.transform(x, y), aoi_wgs84)

        # Use an in-memory dataset to apply rasterio.mask cleanly
        with MemoryFile() as mem:
            with mem.open(**meta) as tmp:
                tmp.write(mosaic)
                clipped, clipped_transform = mask(tmp, [mapping(aoi_in_raster)], crop=True)

                out_meta = meta.copy()
                out_meta.update(
                    {
                        "height": clipped.shape[1],
                        "width": clipped.shape[2],
                        "transform": clipped_transform,
                    }
                )

                with rasterio.open(req.out_tif, "w", **out_meta) as dst:
                    dst.write(clipped)

    finally:
        for s in srcs:
            try:
                s.close()
            except Exception:
                pass

    return req.out_tif



#################################################
## Water Extent and Frequency NASA OPERA DATA  ##
#################################################
