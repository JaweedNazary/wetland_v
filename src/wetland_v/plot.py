from __future__ import annotations

import plotly.graph_objects as go


from shapely.geometry import Polygon, MultiPolygon

import rasterio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import griddata

from scipy.ndimage import gaussian_filter
import random
from .sampling import smoother_data

def plot_dem(file_path, target_grids = 600):
    '''
    Plotting the dem using plotly in 3D:
    file_path = path to the dem file
    target_grid = for speeding the plot process we are plotting a target grid instead of actual size of the raster
    '''

    with rasterio.open(file_path) as src: 
        data = src.read(1)
        transform = src.transform
            
    Z = np.asarray(data, dtype=np.float32)
    ny, nx = Z.shape
    
    
    cols = np.arange(nx)
    rows = np.arange(ny)
    
    # Using rasterio Affine, these attributes exist: a, c, e, f
    x = transform.c + (cols + 0.5) * transform.a
    y = transform.f + (rows + 0.5) * transform.e  # often negative for north-up rasters
    
    Xi, Yi = np.meshgrid(x, y)
    
    
    
    
    Z = Z.astype(float, copy=False)
    Z[Z == -9999.0] = np.nan   # <-- your nodata value
    
    
    
    
    # target around ~600x600 (safe for Plotly surface)
    step = max(1, int(max(nx, ny) / target_grids))
    
    Zs = Z[::step, ::step]
    
    # If you have 1D x,y vectors:
    xs = x[::step] if 'x' in globals() and x is not None else np.arange(0, nx, step)
    ys = y[::step] if 'y' in globals() and y is not None else np.arange(0, ny, step)
    
    
    
    fig = go.Figure(go.Surface(
        x=xs,
        y=ys,
        z=Zs,
        colorscale="Earth_r",
        contours=dict(z=dict(show=True, usecolormap=True, project_z=False))
    ))
    
    # Optional: keep z-range sane (ignore any remaining outliers)
    zmin = np.nanpercentile(Zs, 1)
    zmax = np.nanpercentile(Zs, 99)
    fig.update_layout(scene=dict(zaxis=dict(range=[zmin-300, zmax+300])), width=900, height=700)
    
    return fig


def apply_surface_masks(xi, yi, Zi, geometry, ground_xyz):
    """
    Combines:
    - geometry mask
    - point-density mask
    """
    from shapely import vectorized

    Zi_masked = Zi.copy()

    # Geometry mask
    geom_mask = vectorized.contains(geometry, xi, yi)

    # Point density mask (ground points only)
    density_mask = point_density_mask(xi, yi, ground_xyz[:, :2])

    # Combine both
    final_mask = geom_mask & density_mask

    Zi_masked[~final_mask] = np.nan
    return Zi_masked



def point_density_mask(xi, yi, points_xy):
    """
    xi, yi: 2D meshgrid arrays
    points_xy: (N, 2) array of point x,y coordinates

    Returns:
        mask: True where at least one point exists in the grid cell
    """
    # Grid resolution
    dx = xi[0, 1] - xi[0, 0]
    dy = yi[1, 0] - yi[0, 0]

    x_min, x_max = xi.min(), xi.max()
    y_min, y_max = yi.min(), yi.max()

    # Compute grid indices for each point
    ix = ((points_xy[:, 0] - x_min) / dx).astype(int)
    iy = ((points_xy[:, 1] - y_min) / dy).astype(int)

    # Valid indices only
    valid = (
        (ix >= 0) & (ix < xi.shape[1]) &
        (iy >= 0) & (iy < yi.shape[0])
    )

    ix = ix[valid]
    iy = iy[valid]

    # Count points per cell
    counts = np.zeros(xi.shape, dtype=np.int32)
    np.add.at(counts, (iy, ix), 1)

    return counts > 0




def plot_lidar_masked(pts, geometry, plot_vegetation=True, plot_water = True, sample=50000):
    g_points = pts.ground_xyz
    v_points = pts.veg_xyz
    water_points = pts.water_xyz

    xi, yi, Zi = smoother_data(g_points[:, 0], g_points[:, 1], g_points[:, 2])

    # Mask the surface to your geometry
    Zi_masked = apply_surface_masks(
        xi, yi, Zi,
        geometry=geometry,
        ground_xyz=g_points
        )

    # Sample vegetation (guard for small arrays)
    if plot_vegetation and len(v_points) > 0:
        k = min(sample, len(v_points))
        idx = random.sample(range(len(v_points)), k=k)
        selected_veg = v_points[idx]


    # Sample vegetation (guard for small arrays)
    if plot_water and len(water_points) > 0:
        k = min(sample, len(water_points))
        idx = random.sample(range(len(water_points)), k=k)
        selected_water = water_points[idx]



        
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=xi,
            y=yi,
            z=Zi_masked,
            colorscale='Earth_r',
            colorbar=dict(title='Z'),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=False)
            ),
            name="Ground surface (masked)"
        )
    )

    if plot_vegetation and len(v_points) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=selected_veg[:, 0],
                y=selected_veg[:, 1],
                z=selected_veg[:, 2],
                mode='markers',
                marker=dict(color="green", size=0.5, opacity=0.2),
                name='Vegetation points'
            )
        )

    if plot_water and len(water_points) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=selected_water[:, 0],
                y=selected_water[:, 1],
                z=selected_water[:, 2],
                mode='markers',
                marker=dict(color="blue", size=1.0, opacity=0.2),
                name='Water points'
            )
        )

    # Use masked Z range so NaNs don't mess with min/max
    zmin = np.nanmin(Zi_masked)
    zmax = np.nanmax(Zi_masked)

    fig.update_layout(
        title='3D Surface Plot (clipped to geometry)',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis=dict(
                title='Z',
                range=[zmin - 200, zmax + 200]
            )
        ),
        width=700,
        height=600
    )

    return fig



def plot_lidar(pts, plot_vegetation= True):
    g_points = pts.ground_xyz
    v_points = pts.veg_xyz

    xi, yi, Zi = smoother_data(g_points[:, 0], g_points[:, 1], g_points[:, 2])



    idx = random.choices(range(len(v_points)), k=50000)
    selected_veg = v_points[idx]


    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=xi,
            y=yi,
            z=Zi,
            colorscale='Earth_r',
            colorbar=dict(title='Z'),
            contours=dict(
                z=dict(show=True, usecolormap=True, project_z=False)
            )
        )
    )

    if plot_vegetation== True and len(v_points)>0:
        fig.add_trace(
            go.Scatter3d(
                x=selected_veg[:,0],
                y=selected_veg[:,1],
                z=selected_veg[:,2],
                mode='markers',
                marker=dict(
                    color="green",
                    size=0.5,
                    opacity=0.2
                ),
                name='Original points'
            )
        )
    
    
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis=dict(
                title='Z',
                range=[np.nanmin(Zi) - 200, np.nanmax(Zi) + 200]
            )
        ),
        width=700,
        height=600
    )
    
    

    return fig



def plot_features(samples, gdf, method=1, color = "red"):

    f_x = gdf.geometry.x.to_numpy()
    f_y = gdf.geometry.y.to_numpy()

    if method == 1:
        f_z = np.nanmax(gdf.iloc[:, 5:].to_numpy(), axis=1) + 2.0
    else:
        f_z = np.nanmin(gdf.iloc[:, 5:].to_numpy(), axis=1) + 2.0

    # --- create grid ---
    xi = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 200)
    yi = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 200)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata(
        (samples[:, 0], samples[:, 1]),
        samples[:, 2],
        (X, Y),
        method="linear"
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale="Earth_r",
        colorbar=dict(title="Z")
    ))

    fig.add_trace(go.Scatter3d(
        x=f_x,
        y=f_y,
        z=f_z,
        mode="markers",
        marker=dict(size=5, color=color, opacity=0.2, symbol = "circle"),
        name="Original points"
    ))

    fig.update_layout(
        scene=dict(
            zaxis=dict(range=[np.nanmin(Z)-100, np.nanmax(Z)+100])
        ),
        width=700,
        height=600
    )

    return fig

