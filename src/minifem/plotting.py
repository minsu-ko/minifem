import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .material import ThermalMaterial
from .mesh import Mesh1D
from .physics import heat_flow
from .solver import ThermalResult1D


def _temperature_values(result: ThermalResult1D) -> np.ndarray:
    return np.asarray(result.temperature, dtype=float)


def _element_values(mesh: Mesh1D, nodal_values: np.ndarray) -> np.ndarray:
    if nodal_values.shape != (mesh.num_nodes,):
        raise ValueError(
            f"Expected {mesh.num_nodes} nodal values, "
            f"got {nodal_values.size}."
        )
    return 0.5 * (nodal_values[:-1] + nodal_values[1:])


def _axes(ax):
    return plt.subplots()[1] if ax is None else ax


def _bar_axes(ax, height: float):
    ax.set_ylim(-height / 2, height / 2)
    ax.set_yticks([])
    ax.set_xlabel("Position (m)")
    ax.grid(False)


def plot_mesh(mesh: Mesh1D, ax=None, height: float = 1.0):
    """Plot the 1D mesh as a horizontal bar divided at every node."""
    if height <= 0:
        raise ValueError("Bar height must be positive.")

    ax = _axes(ax)
    y0 = -height / 2
    ax.add_patch(
        Rectangle(
            (mesh.nodes[0], y0),
            mesh.nodes[-1] - mesh.nodes[0],
            height,
            fill=False,
            edgecolor="black",
            linewidth=1.5,
        )
    )
    for node in mesh.nodes[1:-1]:
        ax.vlines(node, y0, y0 + height, color="black", linewidth=1.0)

    _bar_axes(ax, height)
    return ax


def plot_temperature(
    mesh: Mesh1D,
    result: ThermalResult1D,
    ax=None,
):
    """Plot the nodal temperature solution against position."""
    ax = _axes(ax)
    temperatures = _temperature_values(result)
    ax.plot(mesh.nodes, temperatures, "-o")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Temperature (K)")
    ax.grid(True, alpha=0.25)
    return ax


def plot_conductivity(
    mesh: Mesh1D,
    material: ThermalMaterial,
    result: ThermalResult1D | None = None,
    temperature=None,
    ax=None,
):
    """Plot element-center thermal conductivity against position."""
    if result is not None and temperature is not None:
        raise ValueError("Provide either result or temperature, not both.")
    if result is not None:
        temperature = result.temperature
    if temperature is None:
        raise ValueError("A result or nodal temperature array is required.")

    temperatures = np.asarray(temperature, dtype=float)
    element_temperatures = _element_values(mesh, temperatures)
    conductivities = np.asarray(
        [material.k(value) for value in element_temperatures],
        dtype=float,
    )
    centers = 0.5 * (mesh.nodes[:-1] + mesh.nodes[1:])

    ax = _axes(ax)
    ax.plot(centers, conductivities, "-o")
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Thermal conductivity (W / m / K)")
    ax.grid(True, alpha=0.25)
    return ax


def plot_heat_flow(
    mesh: Mesh1D,
    result: ThermalResult1D,
    material: ThermalMaterial,
    area: float,
    ax=None,
):
    """Plot Fourier heat flow in each mesh element."""
    temperatures = _temperature_values(result)
    element_temperatures = _element_values(mesh, temperatures)
    flows = np.asarray(
        [
            heat_flow(
                k=material.k(element_temperatures[index]),
                area=area,
                length=mesh.element_length(index),
                temperature_left=temperatures[index],
                temperature_right=temperatures[index + 1],
            )
            for index in range(mesh.num_elements)
        ],
        dtype=float,
    )
    centers = 0.5 * (mesh.nodes[:-1] + mesh.nodes[1:])

    ax = _axes(ax)
    ax.plot(centers, flows, "-o")
    flow_scale = max(float(np.max(np.abs(flows))), 1.0)
    flow_span = float(np.ptp(flows))
    if flow_span <= max(flow_scale * 1e-10, 1e-12):
        center = float(np.mean(flows))
        margin = max(abs(center) * 0.05, 1e-12)
        ax.set_ylim(center - margin, center + margin)
    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Heat flow (W)")
    ax.grid(True, alpha=0.25)
    return ax