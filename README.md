# minifem

`minifem` is a small Python library for solving one-dimensional, steady-state
thermal conduction problems with the finite element method (FEM).

It supports:

- constant or temperature-dependent thermal conductivity;
- thermal conductivity loaded from a temperature/property CSV file;
- electrical conductivity converted to thermal conductivity with the
  Wiedemann-Franz law;
- temperature, conductivity, and heat-flow plots.

## Installation

Install the package in editable mode from the project directory:

```bash
python -m pip install -e .
```

Run the tests with:

```bash
python -m pytest
```

## The physical problem

The solver models a bar along the $x$ direction. Its cross-sectional area is
constant, and the temperatures at the two ends are prescribed:

$$
T(0) = T_\mathrm{left}, \qquad
T(L) = T_\mathrm{right}.
$$

For steady one-dimensional conduction without internal heat generation,
energy conservation is

$$
\frac{d}{dx}\left(k(T) A \frac{dT}{dx}\right) = 0,
$$

where:

- $T$ is temperature in K;
- $x$ is position in m;
- $A$ is cross-sectional area in m$^2$;
- $k(T)$ is thermal conductivity in W/(m K).

The heat flow is given by Fourier's law:

$$
Q = -k(T) A \frac{dT}{dx}.
$$

The sign convention is that positive heat flow is in the positive $x$
direction.

## FEM algorithm, step by step

### 1. Create a mesh

The bar is divided into nodes. Neighboring nodes form two-node linear
elements:

```python
mesh = minifem.Mesh1D.linspace(
	start=0.0,
	stop=0.1,
	num=51,
)
```

With `num=51`, there are 51 nodes and 50 elements. More nodes give a finer
approximation of the temperature field, but also increase the linear system
that must be solved.

### 2. Approximate the temperature inside each element

For an element with end nodes $i$ and $j$, FEM approximates the temperature by
linear interpolation:

$$
T(x) \approx N_i(x) T_i + N_j(x) T_j,
$$

where $T_i$ and $T_j$ are the unknown nodal temperatures. The implementation
uses the average nodal temperature as the representative element temperature:

$$
T_e = \frac{T_i + T_j}{2}.
$$

This representative temperature is used to evaluate $k(T_e)$.

### 3. Build each element matrix

For a linear element of length $\ell_e$, constant area $A$, and conductivity
$k_e$, the thermal element matrix is

$$
K_e = \frac{k_e A}{\ell_e}
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}.
$$

This matrix describes how the element's two nodal temperatures produce heat
flow into and out of the element.

### 4. Assemble the global matrix

The element matrices are added into a global matrix $K$. Elements sharing a
node contribute to the same row and column. The resulting system is

$$
K T = F,
$$

where $T$ contains all nodal temperatures and $F$ contains prescribed heat
loads. This library has no applied internal heat loads, so $F$ starts as zero.

### 5. Apply the endpoint temperatures

The solver requires temperature boundary conditions at node `0` and node
`mesh.num_nodes - 1`:

```python
boundary_temperatures = {
	0: 300.0,
	mesh.num_nodes - 1: 50.0,
}
```

These are Dirichlet boundary conditions. The corresponding matrix rows and
columns are replaced so the endpoint temperatures are enforced exactly. The
known boundary contributions are moved to the right-hand side before the
columns are cleared.

### 6. Iterate when conductivity depends on temperature

If $k$ is temperature-dependent, the matrix itself depends on the unknown
temperature. The solver handles this by fixed-point iteration:

1. Start with a linear temperature profile between the two boundary values.
2. Evaluate conductivity in each element using its average temperature.
3. Assemble the global matrix.
4. Apply the endpoint temperatures.
5. Solve the resulting linear system.
6. Compare the new and previous temperature vectors.
7. Repeat until
   $\max_i |T_i^{new} - T_i^{old}| < \text{tolerance}$.

The default tolerance is `1e-8`, and the default maximum number of iterations
is `100`. The returned result reports whether convergence was reached and how
many iterations were used.

### 7. Calculate heat flow

After convergence, the solver evaluates Fourier's law on the first element:

$$
Q \approx -k(T_e) A
\frac{T_j - T_i}{\ell_e}.
$$

For a converged steady solution without internal heat generation, this heat
flow should be essentially the same in every element. Small differences can
come from iteration tolerance and numerical roundoff.

## Material models

### Direct thermal conductivity

For a known conductivity function:

```python
material = minifem.ThermalMaterial(
	lambda temperature: 10.0
)
```

For temperature-dependent data stored in a CSV file, use two columns in
strictly increasing temperature order:

```csv
temperature,thermal_conductivity
4,420
100,410
200,395
300,380
```

Then load it with:

```python
material = minifem.ThermalMaterial.from_csv(
	"data/constantan_k.csv"
)
```

Values between rows are linearly interpolated. Temperatures outside the CSV
range raise `ValueError`; the library does not silently extrapolate.

### Wiedemann-Franz material

For a metal where electrical conductivity data is available, use the
Wiedemann-Franz law:

$$
k(T) = L_0 \sigma(T) T,
$$

where $\sigma$ is electrical conductivity in S/m and $L_0$ is the Lorenz
number. The default is

$$
L_0 = 2.44 \times 10^{-8}\ \mathrm{W\,\Omega\,K^{-2}}.
$$

The CSV then contains electrical conductivity rather than thermal
conductivity:

```csv
temperature,electrical_conductivity
4,57769000
100,52727000
200,48333000
300,44615000
```

Load it with:

```python
material = minifem.WiedemannFranzMaterial.from_csv(
	"data/electrical_conductivity.csv"
)
```

The electrical conductivity is linearly interpolated first. The interpolated
value is then converted to thermal conductivity through the Wiedemann-Franz
law.

## Complete example

```python
import minifem

length = 0.1
area = 1e-6
mesh = minifem.Mesh1D.linspace(0.0, length, 51)
material = minifem.ThermalMaterial.from_csv(
	"data/constantan_k.csv"
)

result = minifem.solve_thermal_1d(
	mesh=mesh,
	material=material,
	area=area,
	boundary_temperatures={
		0: 300.0,
		mesh.num_nodes - 1: 50.0,
	},
)

print(result.converged)
print(result.temperature)
print(result.heat_flow)
```

The plotting helpers can be used with the result:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3)
minifem.plot_temperature(mesh, result, ax=axes[0])
minifem.plot_conductivity(mesh, material, result, ax=axes[1])
minifem.plot_heat_flow(mesh, result, material, area, ax=axes[2])
plt.show()
```

## Limitations

This is intentionally a small 1D solver. It currently assumes:

- steady-state conduction;
- one spatial dimension;
- constant cross-sectional area;
- no internal heat generation;
- prescribed temperatures at both mesh endpoints;
- positive thermal conductivity;
- linear two-node elements.