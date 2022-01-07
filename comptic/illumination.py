from .constants import numpy as np


def plane_wave_spatial(
    shape,
    source_position=(0, 0, 50e-3),
    camera_pixel_size=6.5e-6,
    illumination_wavelength=0.55e-6,
    objective_magnification=20,
    system_magnification=1,
    center=True,
    **kwargs
):
    """Plane wave creation"""
    from .util import grid
    from .ledarray import cart_to_na

    # Calculate effective pixel size
    effective_pixel_size = (
        camera_pixel_size / system_magnification / objective_magnification
    )

    # Generate coordinate system
    y_grid, x_grid = grid(shape, effective_pixel_size, center=center)

    # Convert cart to na
    yz = np.sqrt(source_position[1] ** 2 + (source_position[2]) ** 2)
    xz = np.sqrt(source_position[0] ** 2 + (source_position[2]) ** 2)
    na_y = np.sin(np.arctan(source_position[0] / yz))
    na_x = np.sin(np.arctan(source_position[1] / xz))

    # Generate kx and ky
    ky, kx = na_y / illumination_wavelength, na_x / illumination_wavelength

    # Generate plane wave
    phase = -2 * np.pi * (y_grid * ky + x_grid * kx)
    amplitude = 1
    field = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    # Return
    return field


def spherical_wave_spatial(
    shape,
    source_position=(0, 0, 50e-3),
    camera_pixel_size=6.5e-6,
    illumination_wavelength=0.55e-6,
    objective_magnification=20,
    system_magnification=1,
    center=True,
    **kwargs
):
    """Plane wave creation"""
    from .util import grid

    # Calculate effective pixel size
    effective_pixel_size = (
        camera_pixel_size / system_magnification / objective_magnification
    )

    # Generate coordinate system
    y_grid, x_grid = grid(shape, effective_pixel_size, center=center)

    # Generate radius
    radius = np.sqrt(
        (y_grid - source_position[0]) ** 2
        + (x_grid - source_position[1]) ** 2
        + (source_position[2]) ** 2
    )

    # Generate field
    phase = 2 * np.pi / illumination_wavelength * radius
    amplitude = 1 / radius ** 2
    amplitude /= np.max(amplitude)

    # Created field
    field = amplitude * (np.cos(phase) + 1j * np.sin(phase))

    # Return
    return field

