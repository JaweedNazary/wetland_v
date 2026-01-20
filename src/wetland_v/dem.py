from __future__ import annotations

from typing import Any

try:
    from rasterio.enums import Resampling
except Exception:  # pragma: no cover
    Resampling = None


def downsample_dem(dem: Any, target_shape=(1000, 1000)):
    """
    Downsample a rioxarray/xarray object if it exceeds target_shape.
    Expects dem.rio.width/height and dem.rio.reproject().
    """
    if Resampling is None:
        raise ImportError("rasterio is required for downsample_dem(). Install infiltech-geo[dem].")

    scale_factors = [t / s for t, s in zip(target_shape, dem.shape)]

    if any(factor < 1 for factor in scale_factors):
        new_width = dem.rio.width * scale_factors[0] if scale_factors[0] < 1 else dem.rio.width
        new_height = dem.rio.height * scale_factors[1] if scale_factors[1] < 1 else dem.rio.height

        return dem.rio.reproject(
            dem.rio.crs,
            shape=(int(new_height), int(new_width)),
            resampling=Resampling.bilinear,
        )

    return dem
