
"""
wetland_v

Developed by the University of Missouri–Columbia
Contacts: M. Jaweed Nazary, Dr. Kathleen Trauth

Release: January 2026
Version: v0.0.1
License: MIT
"""


from .aoi import AOI, import_shapefile_to_aoi, aoi_from_geojson_geometry
from .crs import proj_to_3857, gcs_to_proj
from .dem import downsample_dem
from .pdal_pipeline import build_pdal_pipeline
from .usgs_3dep import ThreeDEPIndex, load_3dep_index

__all__ = [
    "AOI",
    "import_shapefile_to_aoi",
    "aoi_from_geojson_geometry",
    "proj_to_3857",
    "gcs_to_proj",
    "downsample_dem",
    "build_pdal_pipeline",
    "ThreeDEPIndex",
    "load_3dep_index",
]

__version__ = "0.1.0"
