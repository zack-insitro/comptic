import jax.numpy as np
from . import util


def prop_kernel_fresnel_fourier(
    shape, pixel_size, wavelength, prop_distance, angle_deg=None, RI=1.0
):
    """
    Creates a fresnel propagation kernel in the Fourier Domain
    :param shape: :class:`list, tuple, np.array`
        Shape of sensor plane (pixels)
    :param pixel_size: :class:`float`
        Pixel size of sensor in spatial units
    :param wavelength: :class:`float`
        Detection wavelength in spatial units
    :param prop_distance: :class:`float`
        Propagation distance in spatial units
    :param angle_deg: :class:`tuple, list, np.array`
        Propagation angle, degrees
    :param RI: :class:`float`
        Refractive index of medium
    """
    assert len(shape) == 2, "Propigation kernel size should be two dimensional!"

    # Determine propagation angle and spatial frequency
    angle = len(shape) * [0.0] if angle_deg is None else np.deg2rad(angle_deg)
    fy_illu, fx_illu = [RI * np.sin(a) / wavelength for a in angle]

    # Generate coordinate system
    fylin = _genLin(shape[0], 1 / pixel_size / shape[0])
    fxlin = _genLin(shape[1], 1 / pixel_size / shape[1])

    # Calculate wavenunmber
    k = (2.0 * np.pi / wavelength) * RI

    prop_kernel = np.exp(1j * k * np.abs(prop_distance)) * np.exp(
        -1j
        * np.pi
        * wavelength
        * np.abs(prop_distance)
        * (
            (fxlin[np.newaxis, :] - fx_illu) ** 2
            + (fylin[:, np.newaxis] - fy_illu) ** 2
        )
    )

    return prop_kernel if prop_distance >= 0 else prop_kernel.conj()


def prop_kernel_fresnel_real(
    shape, pixel_size, wavelength, prop_distance, RI=1.0, position=None
):
    """
    Creates a fresnel propagation kernel in the Real Domain
    :param shape: :class:`list, tuple, np.array`
        Shape of sensor plane (pixels)
    :param pixel_size: :class:`float`
        Pixel size of sensor in spatial units
    :param wavelength: :class:`float`
        Detection wavelength in spatial units
    :param prop_distance: :class:`float`
        Propagation distance in spatial units
    :param RI: :class:`float`
        Refractive index of medium
    :param position: :class:`list, tuple, np.array`
        Position of particle center in spatial units
    """
    assert len(shape) == 2, "Propigation kernel size should be two dimensional!"

    # Parse position input
    position = (0, 0) if (not position or len(position) != 2) else position

    # Generate coordinate system
    ygrid, xgrid = util.grid(shape, pixel_size, offset=position)

    # Divice by a common factor of 1000 to prevent overflow errors
    prop_distance /= 1000.0
    wavelength /= 1000.0
    ygrid /= 1000.0
    xgrid /= 1000.0

    # Calculate wavenunmber
    k = (2.0 * np.pi / wavelength) * RI

    # Generate propagation kernel (real-space)
    rr = xgrid ** 2 + ygrid ** 2

    # Generate propagation kernel
    prop_kernel = (
        np.exp(1j * k * prop_distance)
        / (1j * wavelength * prop_distance)
        * np.exp(1j * k / (2 * prop_distance) * rr)
    )

    # Return
    return prop_kernel


def prop_kernel_rayleigh_spatial(
    shape, pixel_size, wavelength, prop_distance, angle=(0, 0)
):

    # Generate coordinate system
    y_grid, x_grid = util.grid(shape, pixel_size)

    # K scalar
    k_scalar = 2 * np.pi / wavelength

    # Calculate radius function
    radius = np.sqrt(prop_distance ** 2 + y_grid ** 2 + x_grid ** 2)

    # Create kernel
    prop_kernel = (
        k_scalar
        / (2j * np.pi)
        * (np.cos(k_scalar * radius) + 1j * np.sin(k_scalar * radius))
        * (1 + 1j / (k_scalar * radius))
        / radius ** 2
    )

    # Flip if negative
    if prop_distance < 0:
        prop_kernel = np.conj(prop_kernel)

    # Add angle
    if np.any(np.array(angle)):
        prop_kernel *= plane_wave_spatial(shape, pixel_size, wavelength, angle)

    return prop_kernel


def prop_kernel_rayleigh_fourier(
    shape, pixel_size, wavelength, prop_distance, angle=(0, 0)
):

    # Calculate angle offset
    k_shift = np.sin(np.deg2rad(np.asarray(angle))) / wavelength

    # Generate coordinate system
    ky_grid, kx_grid = util.grid(
        shape, 1 / pixel_size / np.asarray(shape), offset=k_shift
    )

    # Generate Squared Coordinate System
    fy2 = ky_grid ** 2
    fx2 = kx_grid ** 2
    fz2 = 1 / wavelength ** 2 - fy2 - fx2

    fz = np.sqrt(fz2)
    prop_kernel = np.exp(2j * np.pi * fz * np.abs(prop_distance))

    # Flip if negative
    if prop_distance < 0:
        prop_kernel = np.conj(prop_kernel)

    return prop_kernel


def prop_kernel_angular_spectrum_fourier(
    shape, pixel_size, wavelength, prop_distance, angle=(0, 0)
):

    # Calculate angle offset
    k_shift = np.sin(np.deg2rad(np.asarray(angle))) / wavelength

    # Generate coordinate system
    ky_grid, kx_grid = util.grid(
        shape, 1 / pixel_size / np.asarray(shape), offset=k_shift
    )

    # Inner argument
    argument = (2 * np.pi / wavelength) ** 2 - kx_grid ** 2 - ky_grid ** 2

    # Calculate the propagating and the evanescent (complex) modes
    tmp = np.sqrt(np.abs(argument))
    kz = np.where(argument >= 0, tmp, 1j * tmp)

    # Generate kernel
    kernel = np.exp(1j * kz * prop_distance)

    # Return
    return kernel
