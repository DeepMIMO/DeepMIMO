"""
Geometry module for DeepMIMO channel generation.

This module provides geometric calculations and transformations needed for MIMO systems:
- Array response calculations
- Antenna array geometry functions
- Field of view constraints
- Angle rotations and transformations
- Steering vector computation

The functions handle both single values and numpy arrays for vectorized operations.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


def _array_response(ant_ind: NDArray, theta: float, phi: float, kd: float) -> NDArray:
    """Calculate the array response vector for given antenna indices and angles.
    
    This function computes the complex array response based on antenna positions and 
    arrival angles using the standard array response formula.
    
    Args:
        ant_ind (NDArray): Array of antenna indices with shape (N,3) containing [x,y,z] positions
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians  
        kd (float): Product of wavenumber k and antenna spacing d
        
    Returns:
        NDArray: Complex array response vector with shape matching ant_ind
    """
    gamma = _array_response_phase(theta, phi, kd)
    return np.exp(ant_ind @ gamma.T)


def _array_response_batch(ant_ind: NDArray, theta: NDArray, phi: NDArray, kd: float) -> NDArray:
    """Calculate array response vectors for multiple users/paths simultaneously.
    
    This is a vectorized version of array_response() that can process multiple users
    and paths at once. It handles NaN values in the angle arrays by masking them out.
    
    Args:
        ant_ind (NDArray): Array of antenna indices with shape (N,3) containing [x,y,z] positions
        theta (NDArray): Elevation angles with shape [batch_size, n_paths] in radians
        phi (NDArray): Azimuth angles with shape [batch_size, n_paths] in radians
        kd (float): Product of wavenumber k and antenna spacing d
        
    Returns:
        NDArray: Complex array response matrix with shape [batch_size, N, n_paths]
            where N is number of antenna elements
            
    Note:
        - NaN values in theta/phi are handled by masking - the corresponding output
          elements will be set to 0
        - The function is optimized for batch processing by avoiding loops
        - Output shape allows easy multiplication with path gains in channel generation
    """
    # Get dimensions
    batch_size, n_paths = theta.shape
    n_ant = len(ant_ind)
    
    # Create mask for valid (non-NaN) angles
    valid_mask = ~np.isnan(theta)  # [batch_size, n_paths]
    
    # Calculate phase components for all valid angles
    gamma = _array_response_phase(theta[valid_mask], phi[valid_mask], kd)  # [n_valid_paths, 3]
    
    # Initialize output array
    result = np.zeros((batch_size, n_ant, n_paths), dtype=np.complex128)
    
    # Create index arrays for valid paths
    batch_idx, path_idx = np.nonzero(valid_mask)
    
    # Calculate array response for valid paths
    valid_responses = np.exp(ant_ind @ gamma.T)  # [N, n_valid_paths]
    
    # Assign valid responses to correct positions in output array
    result[batch_idx, :, path_idx] = valid_responses.T
    
    return result


def _array_response_phase(theta: float, phi: float, kd: float) -> NDArray:
    """Calculate the phase components of the array response.
    
    This function computes the phase terms for each spatial dimension x,y,z
    used in array response calculations.
    
    Args:
        theta (float): Elevation angle in radians
        phi (float): Azimuth angle in radians
        kd (float): Product of wavenumber k and antenna spacing d
        
    Returns:
        NDArray: Array of phase components with shape (N,3) for [x,y,z] dimensions
    """
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T


def _ant_indices(panel_size: Tuple[int, int]) -> NDArray:
    """Generate antenna element indices for a rectangular panel.
    
    This function creates an array of indices representing antenna positions 
    in 3D space for a rectangular antenna panel.
    
    Args:
        panel_size (Tuple[int, int]): Panel dimensions as tuple (Mx, My)
        
    Returns:
        NDArray: Array of antenna indices with shape (N,3) where N is total number of elements
    """
    gamma_x = np.tile(np.arange(1), panel_size[0]*panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T


def _apply_FoV(fov: Tuple[float, float], theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Apply field of view constraints to angles.
    
    This function filters angles based on specified field of view limits
    in both elevation and azimuth directions.
    
    Args:
        FoV (Tuple[float, float]): Field of view limits [horizontal, vertical] in degrees
        theta (numpy.ndarray): Elevation angles in radians
        phi (numpy.ndarray): Azimuth angles in radians
        
    Returns:
        NDArray: Boolean mask indicating which angles are within the field of view
    """
    # Convert angles to [0, 2π] range
    theta = np.mod(theta, 2*np.pi)
    phi = np.mod(phi, 2*np.pi)

    # Convert FoV from degrees to radians
    fov = np.deg2rad(fov)

    # Check if azimuth angle is within horizontal FoV
    path_inclusion_phi = np.logical_or(
        phi <= 0 + fov[0]/2,
        phi >= 2*np.pi - fov[0]/2
    )

    # Check if elevation angle is within vertical FoV
    path_inclusion_theta = np.logical_and(
        theta <= np.pi/2 + fov[1]/2,
        theta >= np.pi/2 - fov[1]/2
    )

    # Combine horizontal and vertical masks
    path_inclusion = np.logical_and(path_inclusion_phi, path_inclusion_theta)

    return path_inclusion


def _apply_FoV_batch(fov: Tuple[float, float], theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Apply field of view constraints to angles in batch.
    
    This function filters angles based on specified field of view limits
    in both elevation and azimuth directions for multiple users at once.
    Uses the same FoV for all users.
    
    Args:
        fov (Tuple[float, float]): Field of view limits [horizontal_fov, vertical_fov] in degrees.
            Single FoV applied to all users.
        theta (numpy.ndarray): Elevation angles [batch_size, n_paths] in radians
        phi (numpy.ndarray): Azimuth angles [batch_size, n_paths] in radians
        
    Returns:
        numpy.ndarray: Boolean mask indicating which angles are within the field of view
            Shape: [batch_size, n_paths]
    """
    # Convert angles to [0, 2π] range - exactly matching original function
    theta = np.mod(theta, 2*np.pi)  # [batch_size, n_paths]
    phi = np.mod(phi, 2*np.pi)      # [batch_size, n_paths]
    
    # Convert FoV from degrees to radians
    fov = np.deg2rad(fov)      # [2,]
    
    # Check if azimuth angle is within horizontal FoV - exactly matching original function
    path_inclusion_phi = np.logical_or(phi <= 0 + fov[0]/2, phi >= 2*np.pi - fov[0]/2)
    
    # Check if elevation angle is within vertical FoV - exactly matching original function
    path_inclusion_theta = np.logical_and(theta <= np.pi/2 + fov[1]/2, theta >= np.pi/2 - fov[1]/2)
    
    # Combine horizontal and vertical masks - exactly matching original function
    path_inclusion = np.logical_and(path_inclusion_phi, path_inclusion_theta)
    
    return path_inclusion  # [batch_size, n_paths]


def _rotate_angles(rotation: Optional[Tuple[float, float, float]], theta: float, 
                 phi: float) -> Tuple[float, float]:
    """Rotate angles based on array rotation.
    
    This function applies array rotation to incoming angles. The rotation is specified
    as Euler angles in the order [alpha, beta, gamma] representing rotations around
    the x, y, and z axes respectively.
    
    Args:
        rotation (tuple): Rotation angles [alpha, beta, gamma] in radians for x,y,z axes
        theta (float): Original elevation angle in radians
        phi (float): Original azimuth angle in radians
        
    Returns:
        tuple: (theta_rot, phi_rot) - Rotated elevation and azimuth angles in radians
        
    Note:
        The rotation is applied in the order: z-axis (gamma), y-axis (beta), x-axis (alpha)
        For no rotation, pass None or [0,0,0]. 
        The function uses a specific formulation for rotation that directly computes
        the final angles without intermediate Cartesian coordinate transformations.
    """
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    if rotation is not None:
        rotation = np.deg2rad(rotation)
    
        sin_alpha = np.sin(phi - rotation[2])
        sin_beta = np.sin(rotation[1])
        sin_gamma = np.sin(rotation[0])
        cos_alpha = np.cos(phi - rotation[2])
        cos_beta = np.cos(rotation[1])
        cos_gamma = np.cos(rotation[0])
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        theta = np.arccos(cos_beta*cos_gamma*cos_theta +
                         sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
        phi = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                      1j*(cos_beta*sin_gamma*cos_theta + 
                          sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
    return theta, phi


def _rotate_angles_batch(rotation: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate angles for multiple paths/users simultaneously.
    
    This is a vectorized version of rotate_angles() that can process multiple angles
    at once. It handles NaN values by masking them out.
    
    Args:
        rotation (ndarray): Rotation angles [alpha, beta, gamma] in radians for x,y,z axes
        theta (ndarray): Original elevation angles with shape [batch_size, n_paths]
        phi (ndarray): Original azimuth angles with shape [batch_size, n_paths]
        
    Returns:
        tuple: (theta_rot, phi_rot) - Rotated elevation and azimuth angles
               Both arrays have shape [batch_size, n_paths]
        
    Note:
        The rotation is applied in the order: z-axis (gamma), y-axis (beta), x-axis (alpha)
        NaN values in input angles are preserved in the output
    """
    is_batched = theta.ndim == 2
    if not is_batched:
        theta = theta[None, :]  # [1, n_paths]
        phi = phi[None, :]      # [1, n_paths]
    
    # Ensure rotation is 2D with shape [batch_size, 3] or [1, 3]
    if rotation.ndim == 1:
        rotation = rotation[None, :]  # [1, 3]
    elif rotation.ndim == 3:
        # Handle case where rotation is [batch_size, 0, 3]
        rotation = rotation.reshape(-1, 3)
    
    # Get batch sizes
    batch_size = theta.shape[0]
    rot_batch_size = rotation.shape[0]
    
    # Broadcast rotation if needed
    if rot_batch_size == 1 and batch_size > 1:
        rotation = np.broadcast_to(rotation, (batch_size, 3))
    
    # Convert to radians
    theta = np.deg2rad(theta)  # [batch_size, n_paths] 
    phi = np.deg2rad(phi)      # [batch_size, n_paths]
    rotation = np.deg2rad(rotation)  # [batch_size, 3]
    
    # Extract rotation angles
    alpha = rotation[:, 0:1]  # [batch_size, 1]
    beta = rotation[:, 1:2]   # [batch_size, 1]
    gamma = rotation[:, 2:3]  # [batch_size, 1]
    
    # Compute trigonometric functions - exactly matching original function
    sin_alpha = np.sin(phi - gamma)    # phi - gamma, matches original
    sin_beta = np.sin(beta)            # beta, matches original
    sin_gamma = np.sin(alpha)          # alpha, matches original
    cos_alpha = np.cos(phi - gamma)    # phi - gamma, matches original
    cos_beta = np.cos(beta)            # beta, matches original
    cos_gamma = np.cos(alpha)          # alpha, matches original
    
    sin_theta = np.sin(theta)  # [batch_size, n_paths]
    cos_theta = np.cos(theta)  # [batch_size, n_paths]
    
    # Compute rotated angles using the same formulation as original function
    theta_rot = np.arccos(cos_beta*cos_gamma*cos_theta +
                         sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha))
    
    phi_rot = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta +
                      1j*(cos_beta*sin_gamma*cos_theta + 
                          sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha)))
    
    # Convert back to degrees (not needed)
    # theta_rot = np.rad2deg(theta_rot)  # [batch_size, n_paths]
    # phi_rot = np.rad2deg(phi_rot)      # [batch_size, n_paths]
    # phi_rot = np.mod(phi_rot, 360)     # [batch_size, n_paths]

    # Return angles in radians to match original function
    return (theta_rot[0] if not is_batched else theta_rot,
            phi_rot[0] if not is_batched else phi_rot)


def steering_vec(array: NDArray, phi: float = 0, theta: float = 0, spacing: float = 0.5) -> NDArray:
    """Calculate the steering vector for an antenna array.
    
    This function computes the normalized array response vector for a given array
    geometry and steering direction.
    
    Args:
        array (NDArray): Array of antenna positions
        phi (float): Azimuth angle in degrees. Defaults to 0.
        theta (float): Elevation angle in degrees. Defaults to 0.
        spacing (float): Antenna spacing in wavelengths. Defaults to 0.5.
        
    Returns:
        NDArray: Complex normalized steering (array response) vector
    """
    idxs = _ant_indices(array)
    resp = _array_response(idxs, phi*np.pi/180, theta*np.pi/180 + np.pi/2, 2*np.pi*spacing)
    return resp / np.linalg.norm(resp)