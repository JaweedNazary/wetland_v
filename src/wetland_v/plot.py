from __future__ import annotations

import plotly.graph_objects as go


from shapely.geometry import Polygon, MultiPolygon
from shapely import vectorized

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

    
def mask_grid_by_geometry(xi, yi, Zi, geometry):
    """
    xi, yi, Zi are 2D arrays (meshgrid-style).
    geometry is a shapely Polygon or MultiPolygon.
    Returns Zi_masked where values outside geometry are NaN.
    """
    # vectorized.contains expects 1D arrays; pass the grid directly (it supports ndarray)
    inside = vectorized.contains(geometry, xi, yi)  # shape == Zi.shape
    Zi_masked = Zi.copy()
    Zi_masked[~inside] = np.nan
    return Zi_masked

def plot_lidar_masked(pts, geometry, plot_vegetation=True, veg_sample=50000):
    g_points = pts.ground_xyz
    v_points = pts.veg_xyz

    xi, yi, Zi = smoother_data(g_points[:, 0], g_points[:, 1], g_points[:, 2])

    # Mask the surface to your geometry
    Zi_masked = mask_grid_by_geometry(xi, yi, Zi, geometry)

    # Sample vegetation (guard for small arrays)
    if plot_vegetation and len(v_points) > 0:
        k = min(veg_sample, len(v_points))
        idx = random.sample(range(len(v_points)), k=k)
        selected_veg = v_points[idx]

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

