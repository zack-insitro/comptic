# General utilities
from .constants import numpy as np


def is_iterable(obj):
    try:
        for _ in obj:
            return True
    except Exception:
        return False


def is_array(obj):
    return "array" in str(obj)


def is_double_precision(dtype_or_array):
    if is_array(dtype_or_array):
        if is_complex(dtype_or_array):
            return "128" in str(dtype_or_array.dtype)
        else:
            return "64" in str(dtype_or_array.dtype)
    else:
        if is_complex(dtype_or_array):
            return "128" in str(dtype_or_array)
        else:
            return "64" in str(dtype_or_array)


def is_complex(dtype_or_array):
    if is_array(dtype_or_array):
        return "complex" in str(dtype_or_array.dtype)
    else:
        return "complex" in str(dtype_or_array)


def make_complex(dtype_or_array):
    """Makes a datatype or array complex-valued."""

    if is_array(dtype_or_array):
        return 0 * 1j + dtype_or_array
    else:
        return (1j * np.ones(1, dtype=dtype_or_array)).dtype


def precision(x, for_sum=False):
    """
    This function returns the precision of a given datatype using a comporable numpy array
    """
    if not for_sum:
        return np.finfo(x.dtype).eps
    else:
        return np.finfo(x.dtype).eps * x.size


def grid(shape, scale=1, offset=None, center=True, dtype=None):
    """
    MATLAB-style meshgrid operator. Takes a shape and scale and produces a list of coordinate grids.

    Parameters
    ----------
    shape: list, tuple
        The desired shape of the grid
    scale: list, tuple, int
        Optinal. The scale of the grid. If provided as an integer, provides the
        same scale across all axes. If provided as a list or tuple, must be of
        the same length as shape
    offset: list, tuple, int
        Optinal. Offset of the grid. If provided as an integer, provides the
        same offset across all axes. If provided as a list or tuple, must be of
        the same length as shape.
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    list:
        List of arrays with provided backend and dtype corresponding to
        coordinate systems along each dimension.

    """
    from . import constants

    # Parse scale operation
    if not is_iterable(scale):
        scale = [scale] * len(shape)

    # Parse offset operation
    if offset is None:
        offset = [0] * len(shape)

    # Parse dtype
    dtype = dtype if dtype is not None else constants.DEFAULT_DTYPE

    # Check dimensions
    assert len(shape) == len(scale)
    assert len(shape) == len(offset)

    # Generate axis vectors
    if center:
        axis_vector_list = [
            np.expand_dims(
                (np.arange(_shape, dtype=dtype) - (_shape // 2)) * _scale - _offset,
                axis=[ax for ax in range(len(shape)) if ax != axis],
            )
            for (axis, _shape), _scale, _offset in zip(enumerate(shape), scale, offset)
        ]
    else:
        axis_vector_list = [
            np.expand_dims(
                np.arange(_shape, dtype=dtype) * _scale - _offset,
                axis=[ax for ax in range(len(shape)) if ax != axis],
            )
            for (axis, _shape), _scale, _offset in zip(enumerate(shape), scale, offset)
        ]

    # Initialize grids
    grid_list = [np.ones(shape) * axis_vector for axis_vector in axis_vector_list]

    # Return
    return grid_list


def get_memory_usage():
    """Return memory usage for CPU and GPU in MB"""
    memory_usage_mb = {}

    # Get CPU memory used
    import psutil

    process = psutil.Process(os.getpid())
    memory_usage_mb["cpu"] = process.memory_info().rss / 1024 / 1024

    # Get GPU memory if gpu is available
    if "arrayfire" in config.valid_backends:
        memory_usage_mb["gpu"] = (
            arrayfire.device_mem_info()["alloc"]["bytes"] / 1024 / 1024
        )
    else:
        memory_usage_mb["gpu"] = 0
    return memory_usage_mb


def gradient_check(
    forward, gradient, size, dtype, backend, eps=1e-4, step=1e-3, x=None, direction=None
):
    """
    Check a gradient function using a numerical check.

    Parameters
    ----------
    forward : function
        The forward function of the operator
    gradient : function
        The gradient function of the operator
    size : tuple
        The size of the input to the forward and gradient methods
    dtype : string
        The datatype of the input, as expressed as a string
    backend : string
        The backend of the input
    eps : scalar
        The precision to which the gradient comparison will be held
    x : array-like
        Optional. Array to use for testing the gradient
    direction : array-like
        Optional. Direction to use for testing gradinet

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    # Make sure step is of correct datatype
    step = np.float64(step)

    # Generate x
    if x is None:
        x = randu(size, dtype, backend)
        x[x == 0] = 0.1
    else:
        assert size(x) == shape, (
            "Size of provided input %s does not equal shape of operator %s"
            % (size(x), shape)
        )
        assert getDatatype(x) == dtype

    # Generate direction
    if direction is None:
        # Pick random directions until we get some change in the objective function
        direction = randu(size, dtype, backend)
    else:
        assert size(direction) == shape, (
            "Size of provided direction %s does not equal shape of operator %s"
            % (size(direction), shape)
        )
        assert getDatatype(direction) == dtype

    # Calculate an approximate gradient
    approx_gradient = (
        forward(x + step * direction) - forward(x - step * direction)
    ) / (2 * step)

    # Calculate Gradient to test (override warnings from vec() call)
    g = matmul(
        transpose(vec(gradient(x), no_warnings=True), hermitian=True),
        vec(direction, no_warnings=True),
    )  # dot product

    if not isComplex(g):
        error = sum(abs(g - approx_gradient)) / sum(abs(g + approx_gradient))
    else:
        error = sum(abs(real(g) - approx_gradient)) / sum(abs(g + approx_gradient))

    # Check error
    assert error < eps, "Gradient was off by %.4e (threshold is %.4e)" % (error, eps)

    # Return Error
    return error


def set_byte_order(x, new_byte_order):
    """
    This function sets the byte order of an array
    """
    if new_byte_order.lower() == "f":
        return np.asfortranarray(x)
    elif new_byte_order.lower() == "c":
        return np.ascontiguousarray(x)
    else:
        raise ValueError("Invalid byte order %s" % new_byte_order)

