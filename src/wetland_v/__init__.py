
"""
wetland_v

Developed by the University of Missouri–Columbia
Contacts: M. Jaweed Nazary, Dr. Kathleen Trauth

Release: January 2026
Version: v0.0.1
License: MIT
"""



from .plot import plot_lidar, plot_features

# bring selected “core” things from crawl_trace
from .crawl_trace import (
    AOI,
    ThreeDEPIndex,
    LidarPoints,
    CrawlTraceConfig,
    CrawlTraceResult,
    crawl_trace,     # if you have it
    proj_to_3857,
    gcs_to_proj,
    get_available_years,
    get_lidar_points_around_geometry_3857
)


# Optional backwards-compatible alias
Crawl_Trace = crawl_trace  # remove if you don't want this name


__all__ = [

    # plot
    "plot_lidar",
    "plot_features",

    # core
    "AOI",
    "ThreeDEPIndex",
    "LidarPoints",
    "CrawlTraceConfig",
    "CrawlTraceResult",
    "crawl_trace",
    "Crawl_Trace",
    "proj_to_3857",
    "gcs_to_proj",
    "get_available_years",
    "get_lidar_points_around_geometry_3857",
]
__version__ = "0.1.0"
