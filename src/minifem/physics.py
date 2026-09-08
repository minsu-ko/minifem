import numpy as np


def thermal_element_matrix(k: float, area: float, length: float) -> np.ndarray:
    """
    Return the 2x2 thermal stiffness matrix for a 1D linear element.

    Parameters
    ----------
    k : float
        Thermal conductivity [W / m / K].
    area : float
        Cross-sectional area [m^2].
    length : float
        Element length [m].
    """
    if length <= 0:
        raise ValueError("Element length must be positive.")

    factor = k * area / length

    return factor * np.array([
        [1.0, -1.0],
        [-1.0, 1.0],
    ])


def heat_flow(k: float, area: float, length: float,
              temperature_left: float,
              temperature_right: float) -> float:
    """
    Calculate heat flow through a 1D element using Fourier's law.

    Positive heat flow corresponds to heat flowing in the +x direction.
    """
    return -k * area * (temperature_right - temperature_left) / length