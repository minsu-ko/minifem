from dataclasses import dataclass
import numpy as np


@dataclass
class AWG28:
    diameter: float = 0.321e-3 # meters
    radius = diameter / 2
    area = np.pi * radius ** 2

@dataclass
class AWG30:
    diameter: float = 0.255e-3 # meters
    radius = diameter / 2
    area = np.pi * radius ** 2

@dataclass
class AWG32:
    diameter: float = 0.202e-3 # meters
    radius = diameter / 2
    area = np.pi * radius ** 2
