import numpy as np
import pytest

import minifem


def test_constant_k_thermal_1d():

    length = 0.1
    area = 1e-4
    k = 10.0

    mesh = minifem.Mesh1D.linspace(
        start=0.0,
        stop=length,
        num=11,
    )

    material = minifem.ThermalMaterial(
        lambda T: k
    )

    result = minifem.solve_thermal_1d(
        mesh=mesh,
        material=material,
        area=area,
        boundary_temperatures={
            0: 300.0,
            10: 4.0,
        },
    )

    expected_Q = k * area / length * (300.0 - 4.0)

    assert result.converged
    assert np.isclose(
        result.heat_flow,
        expected_Q,
        rtol=1e-8,
    )


def test_temperature_dependent_k():

    length = 0.1
    area = 1e-4

    k0 = 1.0
    k1 = 0.01

    material = minifem.ThermalMaterial(
        lambda T: k0 + k1 * T
    )

    mesh = minifem.Mesh1D.linspace(
        start=0.0,
        stop=length,
        num=101,
    )

    Th = 300.0
    Tc = 4.0

    result = minifem.solve_thermal_1d(
        mesh=mesh,
        material=material,
        area=area,
        boundary_temperatures={
            0: Th,
            100: Tc,
        },
    )

    expected_Q = (
        area / length
        * (
            k0 * (Th - Tc)
            + 0.5 * k1 * (Th**2 - Tc**2)
        )
    )

    assert result.converged

    assert np.isclose(
        result.heat_flow,
        expected_Q,
        rtol=1e-4,
    )


def test_wiedemann_franz_material_uses_electrical_conductivity():

    lorenz_number = 2.44e-8
    electrical_conductivity = 5.0e7
    temperature = 300.0

    material = minifem.WiedemannFranzMaterial(
        electrical_conductivity=lambda T: electrical_conductivity,
        lorenz_number=lorenz_number,
    )

    expected_k = lorenz_number * temperature * electrical_conductivity

    assert np.isclose(material.k(temperature), expected_k)


def test_material_csv_interpolates_thermal_conductivity(tmp_path):

    path = tmp_path / "thermal.csv"
    path.write_text(
        "temperature,thermal_conductivity\n"
        "100,10\n"
        "300,30\n"
    )

    material = minifem.ThermalMaterial.from_csv(path)

    assert material.k(200.0) == 20.0


def test_wiedemann_franz_csv_interpolates_electrical_conductivity(
    tmp_path,
):

    path = tmp_path / "electrical.csv"
    path.write_text(
        "temperature,electrical_conductivity\n"
        "100,10\n"
        "300,30\n"
    )

    material = minifem.WiedemannFranzMaterial.from_csv(path)

    assert np.isclose(material.k(200.0), 2.44e-8 * 200.0 * 20.0)


def test_material_csv_rejects_temperature_outside_data(tmp_path):

    path = tmp_path / "thermal.csv"
    path.write_text("100,10\n300,30\n")
    material = minifem.ThermalMaterial.from_csv(path)

    with pytest.raises(ValueError, match="outside the CSV range"):
        material.k(50.0)


def test_thermal_solver_requires_mesh_endpoint_boundaries():

    mesh = minifem.Mesh1D.linspace(0.0, 0.1, 51)
    material = minifem.ThermalMaterial(lambda temperature: 10.0)

    with pytest.raises(ValueError, match="nodes 0 and 50"):
        minifem.solve_thermal_1d(
            mesh=mesh,
            material=material,
            area=1e-4,
            boundary_temperatures={0: 300.0, 10: 50.0},
        )


def test_thermal_solver_refines_without_changing_constant_k_result():

    mesh = minifem.Mesh1D.linspace(0.0, 0.1, 51)
    material = minifem.ThermalMaterial(lambda temperature: 10.0)
    result = minifem.solve_thermal_1d(
        mesh=mesh,
        material=material,
        area=1e-4,
        boundary_temperatures={0: 300.0, 50: 50.0},
    )

    expected_heat_flow = 10.0 * 1e-4 / 0.1 * (300.0 - 50.0)

    assert result.converged
    assert np.all(np.diff(result.temperature) < 0.0)
    assert np.isclose(result.heat_flow, expected_heat_flow, rtol=1e-8)