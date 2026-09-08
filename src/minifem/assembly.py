import numpy as np

from .element import ThermalElement1D
from .mesh import Mesh1D
from .material import ThermalMaterial


def assemble_thermal_matrix(
    mesh: Mesh1D,
    material: ThermalMaterial,
    area: float,
    temperature: np.ndarray,
) -> np.ndarray:
    """
    Assemble the global thermal stiffness matrix.
    """

    K = np.zeros((mesh.num_nodes, mesh.num_nodes))

    for e in range(mesh.num_elements):
        i, j = mesh.element_nodes(e)
        length = mesh.element_length(e)

        element = ThermalElement1D(
            node_left=i,
            node_right=j,
            length=length,
            area=area,
            material=material,
        )

        # Representative temperature of the element
        T_element = 0.5 * (temperature[i] + temperature[j])

        Ke = element.stiffness_matrix(T_element)

        # Assembly
        K[i, i] += Ke[0, 0]
        K[i, j] += Ke[0, 1]
        K[j, i] += Ke[1, 0]
        K[j, j] += Ke[1, 1]

    return K