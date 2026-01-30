"""
Wetland V
=========
Developed by:
-------
M. Jaweed Nazary,  PhD Candidate
Dr. Kathleen Trauth, PE, PhD
University of Missouri–Columbia
Department of Civil and Environmetal Engineering

January 2026



ATTENUATION OF FLOOD RISK IN THE MISSOURI RIVER FLOODPLAIN:
CHARACTERIZING PROSPECTS FOR STRATEGIC WETLAND ESTABLISHMENT

This package provides utilities to:
- Charactrixe floodplain geomorphology map and trace levees and channel like feature
- Peform a suitability analysis
- Access and query USGS 3DEP LiDAR datasets
- Download and classify LiDAR point clouds
- Detect and trace linear geomorphic features (e.g., levees, channels)
  using a crawl-and-trace algorithm

Typical workflow
----------------
>>> import wetland_v as wv
>>> idx = wv.load_3dep_index()
>>> cfg = wv.CrawlTraceConfig(...)
>>> result = wv.crawl_trace(cfg)

Submodules
----------
crawl_trace
    Core crawl-and-trace algorithm and configuration classes.
sampling
    Internal numerical and geometric helper routines (not user-facing).
plot
    Visualization utilities for LiDAR points and traced features.


"""

from __future__ import annotations
from shapely.geometry import Point
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
    get_lidar_points
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
    "get_lidar_points",
]
__version__ = "0.1.0"
