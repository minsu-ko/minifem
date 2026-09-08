import numpy as np

from .assembly import assemble_thermal_matrix
from .material import ThermalMaterial
from .mesh import Mesh1D
from .physics import heat_flow


class ThermalResult1D:
    def __init__(self, temperature, heat_flow, iterations, converged):
        self.temperature = temperature
        self.heat_flow = heat_flow
        self.iterations = iterations
        self.converged = converged


def solve_thermal_1d(
    mesh: Mesh1D,
    material: ThermalMaterial,
    area: float,
    boundary_temperatures: dict[int, float],
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ThermalResult1D:

    n = mesh.num_nodes

    if set(boundary_temperatures) != {0, n - 1}:
        raise ValueError(
            "Thermal 1D solves require boundary temperatures at nodes "
            f"0 and {n - 1}."
        )

    # Initial guess: linear interpolation between boundary temperatures
    T = np.zeros(n)

    bc_indices = sorted(boundary_temperatures)

    left = bc_indices[0]
    right = bc_indices[-1]

    T[:] = np.linspace(
        boundary_temperatures[left],
        boundary_temperatures[right],
        n,
    )

    for iteration in range(1, max_iterations + 1):

        # Assemble K using the current temperature
        K = assemble_thermal_matrix(
            mesh=mesh,
            material=material,
            area=area,
            temperature=T,
        )

        F = np.zeros(n)

        # Apply Dirichlet boundary conditions
        K_modified = K.copy()
        F_modified = F.copy()

        for node, temperature in boundary_temperatures.items():
            F_modified -= K_modified[:, node] * temperature
            K_modified[node, :] = 0.0
            K_modified[:, node] = 0.0
            K_modified[node, node] = 1.0
            F_modified[node] = temperature

        # Solve K T = F
        T_new = np.linalg.solve(K_modified, F_modified)

        # Check convergence
        error = np.max(np.abs(T_new - T))

        T = T_new

        if error < tolerance:
            break

    converged = error < tolerance

    # Calculate heat flow from the first element
    i, j = mesh.element_nodes(0)
    length = mesh.element_length(0)

    T_element = 0.5 * (T[i] + T[j])
    k = material.k(T_element)

    Q = heat_flow(
        k=k,
        area=area,
        length=length,
        temperature_left=T[i],
        temperature_right=T[j],
    )

    return ThermalResult1D(
        temperature=T,
        heat_flow=Q,
        iterations=iteration,
        converged=converged,
    )