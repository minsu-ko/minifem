import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np


L0 = 2.44e-8  # W Ohm K^-2


def _read_property_csv(path: str | Path, delimiter: str = ","):
    temperatures = []
    values = []

    with Path(path).open(newline="") as csv_file:
        for row in csv.reader(csv_file, delimiter=delimiter):
            if not row or all(not field.strip() for field in row):
                continue

            try:
                temperature = float(row[0])
                value = float(row[1])
            except (IndexError, ValueError):
                if not temperatures:
                    continue
                raise ValueError(
                    f"Invalid data row in material CSV: {row}"
                ) from None

            temperatures.append(temperature)
            values.append(value)

    if len(temperatures) < 2:
        raise ValueError("Material CSV must contain at least two data rows.")

    temperatures = np.asarray(temperatures, dtype=float)
    values = np.asarray(values, dtype=float)

    if not np.all(np.isfinite(temperatures)) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("Material CSV values must be finite.")
    if not np.all(np.diff(temperatures) > 0):
        raise ValueError("CSV temperatures must be strictly increasing.")

    return temperatures, values


def _interpolator(temperatures, values):
    minimum = temperatures[0]
    maximum = temperatures[-1]

    def interpolate(temperature: float) -> float:
        if not minimum <= temperature <= maximum:
            raise ValueError(
                f"Temperature {temperature} is outside the CSV range "
                f"[{minimum}, {maximum}]."
            )
        return float(np.interp(temperature, temperatures, values))

    return interpolate


class ThermalMaterial:
    """
    Thermal material with temperature-dependent thermal conductivity.
    """

    def __init__(self, conductivity: Callable[[float], float]):
        self._conductivity = conductivity

    @classmethod
    def from_csv(cls, path: str | Path, delimiter: str = ","):
        """Build a material from temperature and thermal conductivity data."""
        temperatures, values = _read_property_csv(path, delimiter)
        return cls(_interpolator(temperatures, values))

    def k(self, temperature: float) -> float:
        """Return thermal conductivity at the given temperature."""
        value = self._conductivity(temperature)

        if value <= 0:
            raise ValueError(
                f"Thermal conductivity must be positive, got {value}."
            )

        return value


class WiedemannFranzMaterial(ThermalMaterial):
    """
    Thermal material whose conductivity is calculated
    from electrical conductivity using the Wiedemann-Franz law.

    k(T) = L0 * T * sigma(T)
    """

    def __init__(
        self,
        electrical_conductivity: Callable[[float], float],
        lorenz_number: float = L0,
    ):
        self._electrical_conductivity = electrical_conductivity
        self.lorenz_number = lorenz_number

        super().__init__(self._conductivity)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        lorenz_number: float = L0,
        delimiter: str = ",",
    ):
        """Build a WF material from temperature and electrical conductivity."""
        temperatures, values = _read_property_csv(path, delimiter)
        return cls(
            electrical_conductivity=_interpolator(temperatures, values),
            lorenz_number=lorenz_number,
        )

    def _conductivity(self, temperature: float) -> float:
        sigma = self._electrical_conductivity(temperature)

        if sigma <= 0:
            raise ValueError(
                f"Electrical conductivity must be positive, got {sigma}."
            )

        return self.lorenz_number * temperature * sigma