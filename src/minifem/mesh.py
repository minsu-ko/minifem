import numpy as np


class Mesh1D:
    def __init__(self, nodes):
        self.nodes = np.asarray(nodes, dtype=float)

        if self.nodes.ndim != 1:
            raise ValueError("Nodes must be a 1D array.")

        if len(self.nodes) < 2:
            raise ValueError("At least two nodes are required.")

        if not np.all(np.diff(self.nodes) > 0):
            raise ValueError("Nodes must be strictly increasing.")

    @classmethod
    def linspace(cls, start: float, stop: float, num: int):
        nodes = np.linspace(start, stop, num)
        return cls(nodes)

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_elements(self):
        return len(self.nodes) - 1

    def element_nodes(self, index: int):
        """Return the two node indices of an element."""
        return index, index + 1

    def element_length(self, index: int):
        i, j = self.element_nodes(index)
        return self.nodes[j] - self.nodes[i]