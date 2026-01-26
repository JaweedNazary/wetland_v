from __future__ import annotations
from typing import Any, Dict, Optional, Literal, List


import plotly.graph_objects as go


from shapely.geometry import Polygon, MultiPolygon, box

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds

import json
import re
import requests
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from pystac_client import Client
import planetary_computer as pc
import earthaccess



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
## NASA OPERA Water Extent and Frequency JPL   ##
#################################################



def _pick_band_links(granule, want_suffix: str):
    """
    Filter a granule's data links to only the band we want (e.g., 'B01_WTR.tif' or 'B01_BWTR.tif').
    """
    links = granule.data_links() or []
    return [u for u in links if u.lower().endswith(want_suffix.lower())]


def download_OPERA(aoi_bbox, start, end, product="HLS", out_dir="opera_download"):
    """
    product: "HLS" (DSWx-HLS, B01_WTR) or "S1" (DSWx-S1, B01_BWTR)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    short_name = "OPERA_L3_DSWX-HLS_V1" if product.upper() == "HLS" else "OPERA_L3_DSWX-S1_V1"
    band_suffix = "B01_WTR.tif" if product.upper() == "HLS" else "B01_BWTR.tif"

    # Earthdata login (will prompt / use env vars EARTHDATA_USERNAME/PASSWORD)
    earthaccess.login()

    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=aoi_bbox,           # (min_lon, min_lat, max_lon, max_lat)
        temporal=(start, end),           # ISO strings, e.g. "2024-10-01", "2024-12-01"
    )

    # Collect only the specific band GeoTIFF links we need
    urls = []
    for g in results:
        urls.extend(_pick_band_links(g, band_suffix))

    if not urls:
        raise RuntimeError("No matching OPERA DSWx band files found for that AOI/time range.")

    # Download to disk
    local_files = earthaccess.download(urls, out_dir)
    # earthaccess can return nested lists depending on version; flatten safely:
    flat = []
    for x in local_files:
        if isinstance(x, (list, tuple)):
            flat.extend(x)
        else:
            flat.append(x)

    return [Path(p) for p in flat]


def compute_OPERA(aoi_bbox, files, product="HLS", out_dir="opera_outputs"):
    """
    Computes:
      - flood_frequency.tif (float32, 0..1)
      - flood_extent.tif (uint8, 0/1)
    Also prints extent area (km^2).

    NOTE (minimal assumption): AOI is small enough that all files share the same CRS/grid (common for one MGRS tile).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class codes (from OPERA GIS visualization guide)
    if product.upper() == "HLS":
        water_codes = {1, 2}                 # open water, partial surface water
        invalid_codes = {252, 253, 254, 255} # snow/ice, cloud/shadow, ocean masked, no data
    else:  # "S1"
        water_codes = {1, 7}                 # open water, inundated vegetation
        invalid_codes = {5, 6, 120, 254, 255} # HAND masked, layover/shadow, no data, ocean masked, (255 just in case)

    # Open first file as reference grid/profile
    with rasterio.open(files[0]) as ref:
        ref_crs = ref.crs
        # Reproject bbox (EPSG:4326) -> dataset CRS to window/crop
        aoi_proj = transform_bounds("EPSG:4326", ref_crs, *aoi_bbox, densify_pts=21)
        ref_win = from_bounds(*aoi_proj, transform=ref.transform)
        ref_profile = ref.profile.copy()

        # Read reference crop to get output shape/transform
        ref_arr = ref.read(1, window=ref_win, boundless=True, fill_value=255)
        out_h, out_w = ref_arr.shape
        out_transform = ref.window_transform(ref_win)

    count_valid = np.zeros((out_h, out_w), dtype=np.uint16)
    count_water = np.zeros((out_h, out_w), dtype=np.uint16)

    for fp in files:
        with rasterio.open(fp) as ds:
            # Minimal guard: require same CRS
            if ds.crs != ref_crs:
                continue

            aoi_proj = transform_bounds("EPSG:4326", ds.crs, *aoi_bbox, densify_pts=21)
            win = from_bounds(*aoi_proj, transform=ds.transform)

            arr = ds.read(1, window=win, boundless=True, fill_value=255)

            # Ensure same shape as reference crop (minimal approach)
            if arr.shape != (out_h, out_w):
                continue

            invalid = np.isin(arr, list(invalid_codes))
            valid = ~invalid

            water = valid & np.isin(arr, list(water_codes))

            count_valid += valid.astype(np.uint16)
            count_water += water.astype(np.uint16)

    # Avoid division by zero
    freq = np.zeros_like(count_water, dtype=np.float32)
    ok = count_valid > 0
    freq[ok] = count_water[ok].astype(np.float32) / count_valid[ok].astype(np.float32)

    extent = (count_water > 0).astype(np.uint8)

    # Pixel area (works if CRS units are meters; OPERA DSWx tiles are UTM at 30m)
    pixel_area_m2 = abs(out_transform.a * out_transform.e)
    extent_area_km2 = extent.sum() * pixel_area_m2 / 1e6
    print(f"Flood extent area in AOI (>=1 detection): {extent_area_km2:.3f} km^2")

    # Write outputs
    freq_profile = ref_profile.copy()
    freq_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        height=out_h,
        width=out_w,
        transform=out_transform,
        compress="deflate",
        tiled=True,
    )

    extent_profile = freq_profile.copy()
    extent_profile.update(dtype="uint8", nodata=0)

    freq_path = out_dir / "flood_frequency.tif"
    extent_path = out_dir / "flood_extent.tif"

    with rasterio.open(freq_path, "w", **freq_profile) as dst:
        dst.write(freq, 1)

    with rasterio.open(extent_path, "w", **extent_profile) as dst:
        dst.write(extent, 1)

    return freq_path, extent_path, freq



#################################################
##              FEMA FLOOD MAP                 ##
#################################################


def arcgis_query(url: str, params: Dict[str, Any], timeout: int = 60, retries: int = 4) -> Dict[str, Any]:
    """ArcGIS REST query with a small retry (helps with occasional 500s)."""
    params = dict(params)
    params.setdefault("f", "json")

    last_err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"ArcGIS request failed after retries: {last_err}")


def esri_rings_to_geojson(rings: List[List[List[float]]]) -> Dict[str, Any]:
    """Basic ESRI rings -> GeoJSON Polygon/MultiPolygon (good enough for mapping)."""

    def ring_area(ring):
        area = 0.0
        for i in range(len(ring) - 1):
            x1, y1 = ring[i]
            x2, y2 = ring[i + 1]
            area += (x1 * y2 - x2 * y1)
        return area / 2.0

    outers, holes = [], []
    for r in rings:
        if len(r) < 4:
            continue
        (outers if ring_area(r) < 0 else holes).append(r)

    if not outers:
        outers, holes = rings, []

    if len(outers) == 1:
        coords = [outers[0], *holes]
        return {"type": "Polygon", "coordinates": coords}

    # Multiple outers -> MultiPolygon (holes assignment omitted for simplicity)
    return {"type": "MultiPolygon", "coordinates": [[[o]] for o in outers]}


def make_query_FEMA(aoi) -> List[int]:
    """Step 1: lightweight spatial query that returns only OBJECTIDs."""
    
    # Your bounding box (min_lon, min_lat, max_lon, max_lat)
    MIN_LON = aoi[0]
    MIN_LAT = aoi[1]
    MAX_LON = aoi[2]
    MAX_LAT = aoi[3]
    
    params = {
        "where": "1=1",
        "geometry": f"{MIN_LON},{MIN_LAT},{MAX_LON},{MAX_LAT}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
    }
    data = arcgis_query(FEMA_LAYER_QUERY_URL, params)
    return data.get("objectIds") or []


def fetch_FEMA(object_ids: List[int], chunk_size: int = 150) -> Dict[str, Any]:
    """Step 2: fetch features in chunks to avoid server 500s."""
    out_fields = ",".join(
        [
            "FLD_ZONE",
            "ZONE_SUBTY",
            "SFHA_TF",
            "STATIC_BFE",
            "V_DATUM",
            "DEPTH",
            "LEN_UNIT",
            "DFIRM_ID",
            "FLD_AR_ID",
        ]
    )

    features = []

    # These two help a LOT:
    # - maxAllowableOffset simplifies geometry server-side (in map units; degrees here)
    # - geometryPrecision reduces coordinate precision
    geom_simplify = {
        "maxAllowableOffset": 0.0001,  # ~11m at equator; good for overview maps
        "geometryPrecision": 5,        # ~1 meter-ish in degrees; plenty for display
    }

    for i in range(0, len(object_ids), chunk_size):
        chunk = object_ids[i : i + chunk_size]
        params = {
            "objectIds": ",".join(map(str, chunk)),
            "outFields": out_fields,
            "returnGeometry": "true",
            "outSR": 4326,
            **geom_simplify,
        }
        data = arcgis_query(FEMA_LAYER_QUERY_URL, params)
        feats = data.get("features") or []
        features.extend(feats)

    # Convert ESRI JSON -> GeoJSON FeatureCollection
    gj_features = []
    for f in features:
        attrs = f.get("attributes") or {}
        geom = f.get("geometry") or {}
        rings = geom.get("rings")
        if not rings:
            continue
        gj_features.append(
            {
                "type": "Feature",
                "properties": attrs,
                "geometry": esri_rings_to_geojson(rings),
            }
        )

    return {"type": "FeatureCollection", "features": gj_features}

