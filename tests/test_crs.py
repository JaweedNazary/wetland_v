from shapely.geometry import Point
from lidar_depo.crs import proj_to_3857

def test_proj_to_3857_runs():
    p = Point(-104.99, 39.74)
    g4326, g3857 = proj_to_3857(p, "EPSG:4326")
    assert g4326.is_valid
    assert g3857.is_valid
