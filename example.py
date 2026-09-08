import matplotlib.pyplot as plt
import numpy as np
import minifem
from pathlib import Path
from minifem.wires import AWG28, AWG30, AWG32

length = 0.1
area = AWG28.area

mesh = minifem.Mesh1D.linspace(
    start=0.0,
    stop=length,
    num=51,
)

# conductivity_data = Path(__file__).parent / "data" / ("electrical_conductivity.csv")
# material = minifem.WiedemannFranzMaterial.from_csv(electrical_conductivity_data)

thermal_conductivity_data = Path(__file__).parent / "data" / ("constantan_k.csv")
material = minifem.ThermalMaterial.from_csv(thermal_conductivity_data)

result = minifem.solve_thermal_1d(
    mesh=mesh,
    material=material,
    area=area,
    boundary_temperatures={
        0: 300.0,
            mesh.num_nodes - 1: 50,
    },
)

print("Converged:", result.converged)
print("Iterations:", result.iterations)
print("Heat flow:", result.heat_flow, "W")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(13, 4),
    constrained_layout=True,
)

axes[0].set_title("Temperature")
minifem.plot_temperature(mesh, result, ax=axes[0])

axes[1].set_title("Thermal conductivity")
minifem.plot_conductivity(mesh, material, result, ax=axes[1])

axes[2].set_title("Heat flow")
minifem.plot_heat_flow(mesh, result, material, area, ax=axes[2])

plt.show()