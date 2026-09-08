import numpy as np

from .material import ThermalMaterial
from .physics import thermal_element_matrix


class ThermalElement1D:
    def __init__(
        self,
        node_left: int,
        node_right: int,
        length: float,
        area: float,
        material: ThermalMaterial,
    ):
        self.node_left = node_left
        self.node_right = node_right
        self.length = length
        self.area = area
        self.material = material

    @property
    def node_indices(self):
        return self.node_left, self.node_right

    def stiffness_matrix(self, temperature: float) -> np.ndarray:
        """
        Return element stiffness matrix evaluated at temperature.
        """
        k = self.material.k(temperature)

        return thermal_element_matrix(
            k=k,
            area=self.area,
            length=self.length,
        )