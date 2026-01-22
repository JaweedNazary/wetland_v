from __future__ import annotations
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import random
from shapely.geometry import LineString
import geopandas as gpd
import pandas as pd

from numba import njit, prange 

def random_window_generator(bounds, num_lines=300, line_length=300):
    # num_lines: number of lines to be generated
    # line_length: length of lines to be generated

    # Create an empty list to store the generated lines
    lines_list = []
    
    threshold = line_length / 2
    
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]

    # Generate random lines
    for _ in range(num_lines):
        centroid_x = random.uniform(min_x + threshold, max_x - threshold)
        centroid_y = random.uniform(min_y + threshold, max_y - threshold)
        orientation_angle_degrees = random.uniform(0, 360)

        orientation_angle_radians = np.radians(orientation_angle_degrees)

        line_endpoint_x1 = centroid_x - (line_length / 2) * np.cos(orientation_angle_radians)
        line_endpoint_y1 = centroid_y - (line_length / 2) * np.sin(orientation_angle_radians)
        line_endpoint_x2 = centroid_x + (line_length / 2) * np.cos(orientation_angle_radians)
        line_endpoint_y2 = centroid_y + (line_length / 2) * np.sin(orientation_angle_radians)

        line = LineString([(line_endpoint_x1, line_endpoint_y1), (line_endpoint_x2, line_endpoint_y2)])

        lines_list.append({'geometry': line})

    # Convert the list to a GeoDataFrame using concat
    gdf = gpd.GeoDataFrame(pd.DataFrame(lines_list), geometry='geometry')

    return gdf


def get_feature_direction(point,points, iteration_k, threshold = 5, r = 50):
    
    one_attractor= get_points_within_r(point, points, r = r)

    x_vals = one_attractor.T[0] - point[0]
    y_vals = one_attractor.T[1] - point[1]
    z_vals = one_attractor.T[2] - point[2]
    
    variances = []
    min_var = float('inf')
    var = 0
    
    # set of direction for check
    tethas = np.linspace(0, np.pi, iteration_k)
    d_vals = np.abs(-np.sin(tethas[:, None])*x_vals[None, :]+np.cos(tethas[:, None])*y_vals[None,:])
    # print(d_vals)
    on_the_line_mask = d_vals<=threshold

    line_y= np.tan(tethas[:, None]) *x_vals[None,:]
    
    for tetha_i in range(len(tethas)):
        if len(z_vals[on_the_line_mask[tetha_i]])>0:
            var = 100*np.var(z_vals[on_the_line_mask[tetha_i]])
            variances.append(var)
            
        if var < min_var:
            min_var = var
            best_tetha = tethas[tetha_i]
            

            
    
    #normalize_variances
    if len(variances) > 0:
        variances_n = (variances - np.mean(variances))/(np.max(variances)-np.min(variances))  
    
        #remove the min value
        variances_no_min = np.delete(variances, np.argmin(variances))
        indicator_value_n = 100 *(np.mean(np.delete(variances_n, np.argmin(variances_n)))-np.mean(variances_n))
        #indicator
        indicator_value = np.mean(variances_no_min)-np.mean(variances)
    else:
        indicator_value = 0
        indicator_value_n = 0
        
    
    return best_tetha, indicator_value, indicator_value_n


def get_points_within_r(point_of_attraction, points, r = 10):
    
    '''This code gets you the points from LiDAR surrunding a point of choice'''
    
    center = np.array([point_of_attraction[0], point_of_attraction[1], 0])
    diff = points - center
    distance = np.sqrt(np.square(diff.T[0]) +np.square(diff.T[1]))
    circle_mask = (distance < r)
    
    one_attractor = points[circle_mask]

    return one_attractor


def get_circle_mask(lines, points, radius = 10):
    M = lines.shape[0]  # number of lines

    x1, y1, x2, y2 = lines[:,0], lines[:,1], lines[:,2], lines[:,3]
    
    centers = np.array([(x1+x2)/2, (y1+y2)/2, np.zeros(M)]).T

    
    diff = points[:,None,:] - centers[None,:,:]
    distance = np.sqrt(np.square(diff.T[0]) +np.square(diff.T[1]))

    
    

    circle_mask = (distance < radius)

    
    
    return circle_mask.T, distance.T



def grid_1D_interpolation(x1, y1, x2, y2, points, n_points=101):
    
    elevation_ = []
    distance_ = []

    for i in range(len(points)):
        xi = points[i][0]
        yi = points[i][1]
        zi = points[i][2]

        # Compute projection distance
        tetha = np.arctan((y2 - y1) / (0.000001+x2 - x1))
        alpha = np.arctan((yi - y1) / (0.000001+xi - x1))
        P1i_ = abs(np.sqrt((xi - x1)**2 + (yi - y1)**2) * np.cos(tetha - alpha))

        distance_.append(P1i_)
        elevation_.append(zi)

    # Convert to numpy
    distance_ = np.array(distance_)
    elevation_ = np.array(elevation_)

    # Sort by distance
    sorter = np.argsort(distance_)
    distance_ = distance_[sorter]
    elevation_ = elevation_[sorter]

    # Interpolate onto equal spacing
    x_new = np.linspace(distance_.min(), distance_.max(), n_points)
    z_new = np.interp(x_new, distance_, elevation_)

    return x_new, z_new



def smoother_data(x_lidar, y_lidar, z_lidar):
    
    # Suppose x, y, z are 1D arrays of scattered data
    points = np.column_stack((x_lidar, y_lidar))
    
    # Define grid
    xi = np.linspace(x_lidar.min(), x_lidar.max(), 500)
    yi = np.linspace(y_lidar.min(), y_lidar.max(), 500)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate onto grid
    Zi = griddata(points, z_lidar, (Xi, Yi), method='nearest')
    
    # Smooth the grid
    Zi_smooth = gaussian_filter(Zi, sigma=1)

    return Xi, Yi, Zi_smooth


@njit
def cross_3d(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    result[0] = u[1]*v[2] - u[2]*v[1]
    result[1] = u[2]*v[0] - u[0]*v[2]
    result[2] = u[0]*v[1] - u[1]*v[0]
    return result

@njit
def get_cross_section_mask(lines: np.ndarray, points: np.ndarray, thickness: float = 10) -> np.ndarray:
    M = lines.shape[0]  # number of lines

    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]

    # Create P0, P1, P2, each shape (M, 3)
    P0 = np.column_stack((x1, y1, np.zeros(M)))
    P1 = np.column_stack((x1, y1, np.full(M, 5.0)))
    P2 = np.column_stack((x2, y2, np.zeros(M)))
    
    # ------------------------------------------------------
    # 2) Compute plane normals N = cross(P1 - P0, P2 - P1)
    # ------------------------------------------------------
    U = P1 - P0  # shape (M,3)
    V = P2 - P1  # shape (M,3)
    
    N = np.empty((M, 3), dtype=np.float64)
    for i in prange(M):
        N[i] = cross_3d(U[i], V[i])
    
    normN = np.sqrt(N[:, 0]**2 + N[:, 1]**2 + N[:, 2]**2)  # shape (M,)
    # Avoid dividing by zero for degenerate lines:
    mask_valid = normN > 1e-12
    for i in range(M):
        if not mask_valid[i]:
            N[i, :] = 0.0
    
    unit_N = np.zeros_like(N)
    for i in range(M):
        if mask_valid[i]:
            unit_N[i] = N[i] / normN[i]
    
    # ------------------------------------------------------
    # 3) Compute distance of each point to each plane.
    #    distance[i,j] = (points[j] - P2[i]) • unit_N[i]
    # ------------------------------------------------------
    plane_point = P2  # shape (M,3)
    num_points = points.shape[0]
    distance = np.empty((num_points, M), dtype=np.float64)
    for i in prange(num_points):
        for j in range(M):
            dx = points[i, 0] - plane_point[j, 0]
            dy = points[i, 1] - plane_point[j, 1]
            dz = points[i, 2] - plane_point[j, 2]
            distance[i, j] = dx*unit_N[j, 0] + dy*unit_N[j, 1] + dz*unit_N[j, 2]
    
    # Keep points within thickness/2:
    plane_mask = np.abs(distance) < (thickness / 2)
    
    # ------------------------------------------------------
    # 4) "Between endpoints" check.
    #    Compute unit vector along the line for each segment.
    # ------------------------------------------------------
    D = P2 - P0  # shape (M,3)
    norm_D = np.sqrt(D[:, 0]**2 + D[:, 1]**2 + D[:, 2]**2).reshape(M, 1)
    valid_D = norm_D[:, 0] > 1e-12  # Boolean array for valid vectors
    unit_D = np.zeros_like(D)
    for i in range(M):
        if valid_D[i]:
            unit_D[i] = D[i] / norm_D[i, 0]
    
    # Compute projections on the line direction
    d0 = np.empty((num_points, M), dtype=np.float64)
    d2 = np.empty((num_points, M), dtype=np.float64)
    for i in prange(num_points):
        for j in range(M):
            # V0 = points[i] - P0[j]
            v0x = points[i, 0] - P0[j, 0]
            v0y = points[i, 1] - P0[j, 1]
            v0z = points[i, 2] - P0[j, 2]
            d0[i, j] = v0x * unit_D[j, 0] + v0y * unit_D[j, 1] + v0z * unit_D[j, 2]
            # V2 = points[i] - P2[j]
            v2x = points[i, 0] - P2[j, 0]
            v2y = points[i, 1] - P2[j, 1]
            v2z = points[i, 2] - P2[j, 2]
            d2[i, j] = v2x * unit_D[j, 0] + v2y * unit_D[j, 1] + v2z * unit_D[j, 2]
    
    between_mask = (d0 >= 0.0) & (d2 <= 0.0)
    
    # ------------------------------------------------------
    # 5) Final mask: a point belongs to cross-section if both conditions hold.
    # ------------------------------------------------------
    cross_section_mask = plane_mask & between_mask
    return cross_section_mask






