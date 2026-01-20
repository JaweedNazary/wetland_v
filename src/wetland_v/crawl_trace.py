from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union, List, Optional
import random
import sys
import warnings
import pyproj
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, shape
import numpy as np
from .crs import proj_to_3857, gcs_to_proj
from .usgs_3dep import ThreeDEPIndex
import json
import requests



warnings.simplefilter(action="ignore", category=FutureWarning)


# Optional numba support
try:
    from numba import prange  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False
    prange = range  # fallback



from .usgs_3dep import load_3dep_index
from .lidar import get_lidar_points_around_geometry_3857
from .sampling import get_cross_section_mask  # <-- wherever you put it
from .sampling import random_window_generator  # <-- wherever you put it
from .sampling import grid_1D_interpolation  # <-- wherever you put it
from .sampling import get_feature_direction  # <-- wherever you put it
from .plot import smoother_data  # <-- wherever you put it
from .crs import proj_to_3857, gcs_to_proj, CRSLike
from .pdal_pipeline import build_pdal_pipeline
from .usgs_3dep import ThreeDEPIndex
from .crs import gcs_to_proj



CRSLike = Union[str, pyproj.CRS]


########################################
### Projecttion Parts                ###
########################################


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




#######################################################
##########     AOI and USGS 3DEP Polygons          ####
#######################################################


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

##############################################################
##### LIDAR DATA DOWNLOAD                            #########
##############################################################

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


def _normalize_year(y):
    if y is None:
        return None
    y = int(y)
    if y < 100:   # treat 17 as 2017, 5 as 2005
        return 2000 + y
    return y


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


def get_lidar_points_around_geometry_3857(geom_3857: BaseGeometry,
                                          index: ThreeDEPIndex,
                                          *,
                                          buffer_distance: float = 3.0,
                                          res: float = 2.0,
                                          out_name: str = "sample_line",
                                          out_dir: str | Path = ".",
                                          prefer_year: Optional[int] = None,
                                          debug: bool = False,):
    """
    EPSG:3857-only workflow.

    - Buffers geom (meters)
    - Finds intersecting 3DEP polygons
    - Runs PDAL EPT->LAS
    - Returns Nx3 arrays (x,y,z)

    prefer_year expects a 4-digit year (e.g., 2019).
    """

    prefer_year_n = _normalize_year(prefer_year)


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
        yr = _parse_year_4digit(name)  # returns 2017, 2018, etc.


        if prefer_year_n is not None:
            if yr is None or yr != prefer_year_n:
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
          aoi_wkt,          # extent_epsg3857_wkt (positional is safest)
          datasets,         # usgs_3dep_dataset_names
          res,              # pc_resolution
          filter_noise=False,
          reclassify=False,
          save_pointcloud=True,
          out_crs=3857,
          pc_out_name=str(las_path.with_suffix("")),  # base path without extension
          pc_out_type="las",
          debug=debug,)

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

##############################################################
############# CRAWL TRACE                             ########
##############################################################


def one_step_crawl(
    lines: np.ndarray,
    points: np.ndarray,
    mask: np.ndarray,
    point_of_attraction: np.ndarray,
    line_length: float,
    *,
    method: int = 1,
) -> np.ndarray:
    """
    Move each line segment center to a selected point among points within its mask.

    Parameters
    ----------
    lines : (M,4) array
        Each row: [x1, y1, x2, y2]
    points : (N,3) array
        Point cloud XYZ
    mask : (N,M) bool array
        mask[n,i] True if point n belongs to line i's neighborhood
    point_of_attraction : (M,3) array
        Updated in-place with [x,y,H] where H = max(z)-min(z) for that line's points
    line_length : float
        New segment length
    method : int
        1=max(z), 2=min(z), 3=first point with z>=90th percentile (fallback to max)

    Returns
    -------
    endpoints : (M,5) float array
        [new_x1, new_y1, new_x2, new_y2, H] for moved lines.
        Rows remain 0 if a line had no points.
    """
    M = lines.shape[0]
    endpoints = np.zeros((M, 5), dtype=np.float64)

    for i in prange(M):
        old_center_x = 0.5 * (lines[i, 0] + lines[i, 2])
        old_center_y = 0.5 * (lines[i, 1] + lines[i, 3])

        pts_i = points[mask[:, i]]
        if pts_i.shape[0] == 0:
            continue

        col_z = pts_i[:, 2]
        H = float(np.max(col_z) - np.min(col_z))

        if method == 1:
            idx = int(np.argmax(col_z))
        elif method == 2:
            idx = int(np.argmin(col_z))
        elif method == 3:
            target = np.percentile(col_z, 90)
            idxs = np.where(col_z >= target)[0]
            idx = int(idxs[0]) if idxs.size > 0 else int(np.argmax(col_z))
        else:
            raise ValueError("method must be 1, 2, or 3")

        new_center_x = float(pts_i[idx, 0])
        new_center_y = float(pts_i[idx, 1])

        # Update point_of_attraction (always, if pts exist)
        point_of_attraction[i, 0] = new_center_x
        point_of_attraction[i, 1] = new_center_y
        point_of_attraction[i, 2] = H

        # Build new segment endpoints in direction from old to new center
        dx_c = new_center_x - old_center_x
        dy_c = new_center_y - old_center_y
        if dx_c == 0.0 and dy_c == 0.0:
            continue

        theta = np.arctan2(dy_c, dx_c)
        dx = 0.5 * line_length * np.cos(theta)
        dy = 0.5 * line_length * np.sin(theta)

        endpoints[i, 0] = new_center_x - dx
        endpoints[i, 1] = new_center_y - dy
        endpoints[i, 2] = new_center_x + dx
        endpoints[i, 3] = new_center_y + dy
        endpoints[i, 4] = H

    return endpoints

def Crawl_Trace(location, N, min_height, max_height, window_size, D, r, resolution, method, random_seed, sigma = 2.0):
    """
    Trace levee-like or channel-like linear features from high-resolution LiDAR
    within a 1000 m × 1000 m study tile centered at a user-specified location.

    Developed by M. Jaweed Nazary as part of a PhD dissertation at the
    University of Missouri — Columbia. Work supported by EPA Grant No. CD97790701.
    All rights reserved (University of Missouri — Columbia).
    Last update: October 29, 2025, 13:54.

    The routine automatically extracts LiDAR for the study tile, seeds N random
    starting points, and "crawls" from each seed to detect and trace linear
    landforms. Two crawling modes are available: uphill tracing (for levees and
    ridge-like features) and downhill tracing (for channels and depression-like
    features). For larger study areas, run this function in a tiling loop and
    aggregate results.

    Args:
        location (tuple[float, float]):
            Center coordinate of the study tile in EPSG:3857 (x, y). Units: meters.
            Example: (-10523769.57, 4719384.06)
        N (int):
            Number of random seed points generated inside the 1000 m tile.
            Larger N increases detection coverage at the cost of runtime.
            Example: 100
        min_height (float):
            Minimum relative height (meters) for a candidate landform to be
            considered. Features below this threshold are ignored.
            Example: 1.0
        max_height (float):
            Maximum relative height (meters) for candidate landforms. Features
            above this threshold are ignored.
            Example: 50.0
        window_size (tuple[float, float]):
            Local tracing window dimensions (L, W) in meters: length L and
            width W used when tracing and sampling terrain.
            Example: (100.0, 5.0)
        D (float):
            Lateral offset distance (meters) used in the tracing procedure.
            Example: 20.0
        r (float):
            Neighborhood radius (meters) used to estimate local orientation of
            the landform.
            Example: 20.0
        resolution (int):
            Requested LiDAR processing resolution (point spacing) in meters.
            Common supported values: 1, 2, 5, 10 (depends on available LiDAR and PDAL pipeline).
            Example: 1
        method (int):
            Crawling method selector:
              * 1 — uphill crawling (detect levees / ridges)
              * 2 — downhill crawling (detect channels / depressions)
            Example: 1
        random_seed (int):
            Integer seed for random number generation to make results reproducible.
            Example: 59

    Returns:
        geopandas.GeoDataFrame (recommended):
            GeoDataFrame of traced linear features (LineString) with attributes
            such as `trace_id`, `method`, `avg_height`, `min_height`, `max_height`,
            and optionally a `confidence` score. If the implementation returns a
            different structure, adapt this section accordingly.

    Raises:
        ValueError: if input validation fails (e.g., non-numeric coordinates,
                    min_height > max_height, non-positive window_size or r).
        RuntimeError: if LiDAR extraction (PDAL) fails or required data are unavailable.

    Notes:
        - Coordinates must be provided in EPSG:3857 (units = meters). Reproject
          coordinates before calling this function if needed.
        - The function operates on a 1000 m × 1000 m tile centered at `location`.
          For continuous traces across larger areas, use overlapping tiles when
          running the function in a loop.
        - Choose `resolution` and `window_size` according to the scale of features
          you expect: smaller values capture finer details but increase compute cost.
        - Use the same `random_seed` to reproduce a given run.
        - Visual QA is recommended: overlay traces on the DEM/DSM and compare
          with known datasets (e.g., National Levee Dataset) where available.

    Example:
        >>> location = (-10523769.57, 4719384.06)
        >>> traces = Crawl_Trace(location, N=100, min_height=1.0, max_height=50.0,
                                 window_size=(100.0, 5.0), D=20.0, r=20.0,
                                 resolution=1, method=1, random_seed=59)
    """
    
    
    
    gdf_points = gpd.GeoDataFrame(columns = ['geometry', 'H', 'i_n', 'I_1', 'theta'], geometry='geometry', crs="EPSG:3857")
    gdf_traced_lines = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:3857" )


    random.seed(random_seed) 
        
    file_size = 0
    no_of_points = []
    no_of_g_points = []
    points_cs_for_future_use = []
    xy_123 = []

    # max relative height for selecting features
    year_of_record = []
    x = location[0]
    y = location[1]
    point_geom = Point(x, y)
    index = load_3dep_index()
    print(f'Downloading LiDAR data...')
    pts = get_lidar_points_around_geometry_3857(point_geom, index, buffer_distance = 500, res = resolution, out_name="las")

    g_points = pts.ground_xyz
    all_points = pts.all_xyz
    r_year = pts.most_recent_year
#     print(f'No of Points in las file = {len(points)}, No of classified ground points = {len(g_points)}, Year = {year_}')
    no_of_points.append(len(all_points))
    no_of_g_points.append(len(g_points))
    year_of_record.append(r_year)
    file_size += all_points.nbytes / (1024**2)   # in MB  


    min_x = x-500
    max_x = x+500
    min_y = y-500
    max_y = y+500

    bounds = [min_x, min_y, max_x, max_y]


    # Creating random starting windows for crawling algorithm
    window_gdf = random_window_generator(bounds = bounds, num_lines=N, line_length=window_size[0])

    xi, yi, zi = smoother_data(g_points[:, 0], g_points[:,1], g_points[:,2]) 

    x_long, y_long, z_long = xi.reshape([250000]), yi.reshape([250000]), zi.reshape([250000])
    points = np.vstack([x_long, y_long, z_long]).T


    # points = np.vstack([g_points.x, g_points.y, g_points.z]).T
    
    # Randomly sample indices without replacement
    if len(points)>50000:
        sample_indices = np.random.choice(len(points), size=50000, replace=False)
        sampled_points = points[sample_indices]
    else:
        sampled_points = points


    lines = np.array([list(line.coords[0]) + list(line.coords[1]) for line in window_gdf.geometry])

    #starting with empth set
    point_of_attraction = np.empty((len(lines), 3), dtype=np.float64)

    ######################################################################################
    ## CRAWLING PROCESS to find points of attraction                                     #
    ######################################################################################
    print(f'Crawling on the landscape...')
    while lines.shape[0] > 0:
        mask = get_cross_section_mask(lines, points, thickness=window_size[1])
        lines = lines[:,:4]
        lines = one_step_crawl(lines, points, mask, point_of_attraction,window_size[0], method=method)
        lines = lines[~np.all(lines == 0, axis=1)]
        lines = lines[(lines[:, 4] >= min_height) & (lines[:, 4] <= max_height)]


    point_of_attraction = point_of_attraction[~np.all(point_of_attraction ==0, axis=1)]
    point_of_attraction = point_of_attraction[(point_of_attraction[:,2] >= min_height) & (point_of_attraction[:, 2] <= max_height)]



    directions=[]
    indicators = []
    indicators_norm = []
    f_points = []

    ################################################################################
    ## EXTRACTING GEOMORPHOMETRIC INFORMATION FOR EACH ATTRACTION POINT            #
    ################################################################################

    for i in range(len(point_of_attraction.T[0])):
        ang_, indc, indc_n = get_feature_direction(point_of_attraction[i],points, 10, threshold = 5, r = r)

        if indc>0:
            directions.append(ang_ )
            indicators.append(indc)
            indicators_norm.append(indc_n)
            f_points.append(point_of_attraction[i])




    ##############################################################################
    ## TRACING PROCESS                                                           #
    ##############################################################################
    print(f'Tracing features of interest...')
    offset_distances = [D, -D]      # offset distance, both direction, left and right
    method = method                        # Method 1 traces levees and method 2 traces channels
    L = window_size[0]                           # Lenght at which cross section information will be analyzed

    boundary_ = LineString([(min_x,min_y),
                            (min_x,max_y),
                            (max_x,max_y),
                            (max_x,min_y), 
                            (min_x,min_y)])

    for j in range(len(f_points)):    # f_points are all the attraction points

        no_bins= (50*(1+j))//len(f_points)
        sys.stdout.write(f"\rTracing \033[92m {(no_bins+1)*'▃'}\033[97m{(50-no_bins)*'▃'} \033[92m feature {j+1} out of {len(f_points):0.2f}, \
    {100*(1+j)/len(f_points):0.2f} %")
        sys.stdout.flush()
        
        # def set_offset(start_point, offset_distance, tetha):
        new_point = f_points[j]
        x_s = new_point[0]
        y_s = new_point[1]

        for offset_distance in offset_distances:
            lines_traced = []
            new_point = f_points[j]
            x_s = new_point[0]
            y_s = new_point[1]


            n=0
            ind = 5
            # Tracing upto 100 times in each direction 
            while n<100:
                # extracting feature information
                var = 0
                tetha,ind, ind_n = get_feature_direction(new_point,points, 10, threshold = 5, r = r)

                # if ind<1:
                   # pass
                if ind_n < 2.0:
                    pass
                else:
                    x_s = x_s + offset_distance*np.abs(np.cos(tetha))
                    y_s = y_s + offset_distance*np.abs(np.sin(tetha))


                    x1 = x_s + L * np.cos(tetha+np.pi/2)
                    y1 = y_s + L * np.sin(tetha+np.pi/2)
                    x2 = x_s - L * np.cos(tetha+np.pi/2)
                    y2 = y_s - L * np.sin(tetha+np.pi/2)
                    line = np.array([[x1, y1, x2, y2]], dtype=np.float64)
                    mask = get_cross_section_mask(line, points, thickness = window_size[1])
                    points_line_i = points[mask[:,0]]


                    col_z = points_line_i[:, 2]
                    

                    

                    # crawling to the center of the levee
                    if len(col_z)>0:
                        dis, elv = grid_1D_interpolation(x1, y1, x2, y1, points_line_i, n_points=100)

                        if method == 1:
                            idx = np.argmax(col_z)

                        elif method == 2:
                            idx = np.argmin(col_z)

                        
                        elif method == 3:
                            target = np.percentile(col_z, 90)
                            idxs = np.where(col_z >= target)[0]
                            if len(idxs) == 0:
                                idx = np.argmax(col_z)
                            else:
                                idx = idxs[0]
                        
                        else:
                            raise ValueError("Unknown method")


                        x_s = points_line_i[idx][0]
                        y_s = points_line_i[idx][1] 

                        x1 = x_s + L * np.cos(tetha+np.pi/2)
                        y1 = y_s + L * np.sin(tetha+np.pi/2)
                        x2 = x_s - L * np.cos(tetha+np.pi/2)
                        y2 = y_s - L * np.sin(tetha+np.pi/2)

                        # cross section of the levee
                        x_section = LineString([(x1, y1), (x2, y2)]) 




                        # checking if this is a new cross section or one we aleary have
                        if gdf_traced_lines.geometry.intersects(x_section).any() == False:

                            # checking for H
                            H = np.max(col_z) - np.min(col_z)

                            if H < min_height:
                                break

                            else:
                                lines_traced.append((x_s, y_s))
                                new_point = np.array([x_s, y_s, 0])
                                new_point_geom = Point(x_s, y_s)

                                
                                # --- build dictionary for the new row ---
                                row_dict = {
                                    'geometry': new_point_geom,
                                    'H': H,
                                    'i_n': ind_n,
                                    'I_1': ind,
                                    'theta': tetha
                                }
                            
                                # Add the 100 z-values as z_1 … z_100
                                for i in range(100):
                                    row_dict[f"z_{i+1}"] = elv[i]
        

                                # adding the traced point to the gdf_points
                                new_point_gdf = gpd.GeoDataFrame([row_dict], crs="EPSG:3857")

                                # Concatenate the new GeoDataFrame to the existing one
                                # Only concatenate if new_point_gdf has valid geometries
                                if not new_point_gdf.empty and new_point_gdf.geometry.notna().any(): 
                                    gdf_points = pd.concat([gdf_points, new_point_gdf], ignore_index=True)

                                ##### sTORING THE CROSS SECTION POINTS 
                                line__ = np.array([[x1, y1, x2, y2]], dtype=np.float64)
                                mask = get_cross_section_mask(line__, points, thickness = window_size[1])
                                points_line__ = points[mask[:,0]]
                                points_cs_for_future_use.append(points_line__)
                                xy_123.append([x1, y1, x2, y2])

                                if new_point_geom.within(boundary_.buffer(L)):
                                    break


                        else:
                            break


                n+=1




            if len(lines_traced)>2:
                line_geom = LineString(lines_traced)
                new_line_gdf = gpd.GeoDataFrame({'geometry': [line_geom]}, crs="EPSG:3857")
                new_line_gdf = new_line_gdf.set_geometry(new_line_gdf.geometry.astype('geometry'))

                # Concatenate the new GeoDataFrame to the existing one
                # Only concatenate if new_point_gdf has valid geometries
                if not new_line_gdf.empty and new_line_gdf.geometry.notna().any():
                    gdf_traced_lines = pd.concat([gdf_traced_lines, new_line_gdf], ignore_index=True)
    print(f'Crawling and tracing processes are successfully compeleted.')        
    return gdf_points, gdf_traced_lines, sampled_points


