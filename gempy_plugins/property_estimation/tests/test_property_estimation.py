import numpy as np
import pytest
import pyvista

pyvista.OFF_SCREEN = True
pyvista.BUILDING_GALLERY = True

import gempy as gp
import gstools as gs

from gempy_plugins.property_estimation.conditioning_data import ConditioningData
from gempy_plugins.property_estimation.domains import (
    compute_domains, describe_domains, domain_mask, group_mask, validate_disjoint_groups,
)
from gempy_plugins.property_estimation.kriging import KrigingDomainConfig, run_kriging
from gempy_plugins.property_estimation.neighborhood import Neighborhood
from gempy_plugins.property_estimation.plotting import plot_domains, plot_fault_blocks, plot_property_field
from gempy_plugins.property_estimation.simulation import SimulationDomainConfig, run_simulation

data_path = "https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/"


@pytest.fixture(scope="module")
def geo_model():
    geo_model = gp.create_geomodel(
        project_name="combination",
        extent=[0, 2500, 0, 1000, 0, 1000],
        resolution=[20, 8, 8],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/data/input_data/jan_models/model7_orientations.csv",
            path_to_surface_points=data_path + "/data/input_data/jan_models/model7_surface_points.csv",
        )
    )
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            "Fault_Series": ("fault"),
            "Strat_Series1": ("rock3"),
            "Strat_Series2": ("rock2", "rock1"),
        }
    )
    gp.set_is_fault(geo_model, ["Fault_Series"])
    gp.compute_model(geo_model)
    return geo_model


@pytest.fixture(scope="module")
def domains(geo_model):
    return compute_domains(geo_model)


@pytest.fixture(scope="module")
def conditioning_data(geo_model, domains):
    lith_array, fault_array, _ = domains
    np.random.seed(1)
    n = 200
    data = ConditioningData(
        x=np.random.uniform(0, 2500, n),
        y=np.random.uniform(0, 1000, n),
        z=np.random.uniform(0, 1000, n),
        values=np.random.normal(15, 3, n),
    )
    data.assign_domains(geo_model, lith_array, fault_array)
    return data


def test_compute_domains(geo_model, domains):
    lith_array, fault_array, domain_keys = domains
    resolution = tuple(geo_model.grid.regular_grid.resolution)

    assert lith_array.shape == resolution
    assert fault_array.shape == resolution
    assert len(domain_keys) > 0

    # every domain key actually selects at least one cell, and masks are disjoint
    total_masked = 0
    for domain_key in domain_keys:
        mask = domain_mask(lith_array, fault_array, domain_key)
        assert mask.any()
        total_masked += mask.sum()
    assert total_masked == lith_array.size  # domains partition the whole grid


def test_describe_domains(geo_model, domains):
    _, _, domain_keys = domains
    descriptions = describe_domains(geo_model, domain_keys)

    assert set(descriptions.keys()) == set(domain_keys)
    known_liths = {"rock1", "rock2", "rock3", "basement"}
    for name in descriptions.values():
        assert any(lith in name for lith in known_liths)


def test_plot_domains_colors_share_hue_per_lithology(geo_model, domains):
    lith_array, fault_array, domain_keys = domains
    plotter = plot_domains(geo_model, lith_array, fault_array, domain_keys, show=False)

    from gempy_plugins.property_estimation.plotting import _domain_colors
    colors = _domain_colors(domain_keys)

    # domains sharing a lithology must share hue (but not be identical -- distinct shades)
    import matplotlib.colors as mcolors
    by_lith = {}
    for domain_key in domain_keys:
        lith_id, _ = domain_key
        by_lith.setdefault(lith_id, []).append(colors[domain_key])

    for lith_id, rgb_list in by_lith.items():
        hues = [mcolors.rgb_to_hsv(rgb)[0] for rgb in rgb_list]
        assert max(hues) - min(hues) < 1e-6, f"lith {lith_id} shades should share a hue"
        if len(rgb_list) > 1:
            assert len(set(rgb_list)) == len(rgb_list), f"lith {lith_id} shades should be distinct"

    plotter.close()


def test_plot_fault_blocks_labels_each_block(geo_model, domains):
    lith_array, fault_array, domain_keys = domains
    plotter = plot_fault_blocks(geo_model, fault_array, show=False)

    n_fault_blocks = len(np.unique(fault_array))
    assert len(np.unique(plotter.mesh.cell_data['fault_block'])) == n_fault_blocks
    plotter.close()


def test_plot_property_field_only_shows_computed_domains(geo_model, domains, conditioning_data):
    lith_array, fault_array, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)
    configs = {domain_keys[0]: KrigingDomainConfig(model=model)}
    field = run_kriging(geo_model, conditioning_data, configs)

    plotter = plot_property_field(geo_model, field, show=False)
    mask = domain_mask(lith_array, fault_array, domain_keys[0])
    assert plotter.mesh.n_cells == mask.sum()
    plotter.close()


def test_conditioning_data_assign_and_filter(domains, conditioning_data):
    lith_array, fault_array, domain_keys = domains

    assert conditioning_data.lith_id is not None
    assert conditioning_data.fault_id is not None
    assert len(conditioning_data.lith_id) == len(conditioning_data)

    # every assigned (lith, fault) pair is one of the real domain keys
    assigned_pairs = set(zip(conditioning_data.lith_id.tolist(), conditioning_data.fault_id.tolist()))
    assert assigned_pairs.issubset(set(domain_keys))

    domain_key = domain_keys[0]
    subset = conditioning_data.for_domain(domain_key)
    assert len(subset) <= len(conditioning_data)
    assert np.all(subset.lith_id == domain_key[0])
    assert np.all(subset.fault_id == domain_key[1])


def test_for_domain_without_assign_raises():
    data = ConditioningData(x=[0], y=[0], z=[0], values=[1.0])
    with pytest.raises(RuntimeError):
        data.for_domain((1, 0))


def test_conditioning_data_length_mismatch_raises():
    with pytest.raises(ValueError):
        ConditioningData(x=[0, 1], y=[0], z=[0, 1], values=[1.0, 2.0])


def test_run_kriging_limited_domains(geo_model, domains, conditioning_data):
    lith_array, fault_array, domain_keys = domains
    # a small nugget avoids the well-known ill-conditioning of a smooth, nugget-free
    # Gaussian covariance model at higher point counts
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)

    selected = domain_keys[:2]
    configs = {domain_key: KrigingDomainConfig(model=model) for domain_key in selected}
    field = run_kriging(geo_model, conditioning_data, configs)

    assert set(field.domain_keys) == set(selected)
    assert field.values.shape == lith_array.shape

    populated_mask = np.zeros_like(lith_array, dtype=bool)
    for domain_key in selected:
        populated_mask |= domain_mask(lith_array, fault_array, domain_key)

    assert np.all(np.isnan(field.values[~populated_mask]))
    assert not np.any(np.isnan(field.values[populated_mask]))
    # values should stay in a sane range around the conditioning data (15 +/- 3)
    assert np.nanmin(field.values) > -20
    assert np.nanmax(field.values) < 50


def test_run_kriging_different_settings_per_domain(geo_model, domains, conditioning_data):
    _, _, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)

    configs = {
        domain_keys[0]: KrigingDomainConfig(model=model),
        domain_keys[1]: KrigingDomainConfig(model=model, krige_class=gs.krige.Simple, krige_kwargs={"mean": 15}),
    }
    field = run_kriging(geo_model, conditioning_data, configs)
    assert set(field.domain_keys) == {domain_keys[0], domain_keys[1]}


def test_run_kriging_n_closest_neighborhood(geo_model, domains, conditioning_data):
    lith_array, fault_array, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)
    domain_key = domain_keys[0]

    configs = {domain_key: KrigingDomainConfig(model=model, neighborhood=Neighborhood(mode="n_closest", n=5))}
    field = run_kriging(geo_model, conditioning_data, configs)

    assert field.domain_keys == [domain_key]
    mask = domain_mask(lith_array, fault_array, domain_key)
    assert np.sum(~np.isnan(field.values)) == mask.sum()


def test_run_kriging_missing_conditioning_data_warns(geo_model, domains):
    _, _, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)
    empty_data = ConditioningData(x=[], y=[], z=[], values=[])
    lith_array, fault_array, _ = domains
    empty_data.assign_domains(geo_model, lith_array, fault_array)

    configs = {domain_keys[0]: KrigingDomainConfig(model=model)}
    with pytest.warns(UserWarning):
        field = run_kriging(geo_model, empty_data, configs)
    assert field.domain_keys == []
    assert np.all(np.isnan(field.values))


def test_run_kriging_merged_fault_blocks(geo_model, domains, conditioning_data):
    """Merging both fault blocks of one lithology (e.g. a negligible-offset fault)."""
    lith_array, fault_array, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)

    same_lith_pair = next(
        (a, b) for a in domain_keys for b in domain_keys if a != b and a[0] == b[0]
    )
    group = (same_lith_pair[0], same_lith_pair[1])
    configs = {group: KrigingDomainConfig(model=model)}
    field = run_kriging(geo_model, conditioning_data, configs)

    assert set(field.domain_keys) == set(group)
    expected_mask = group_mask(lith_array, fault_array, group)
    assert np.sum(~np.isnan(field.values)) == expected_mask.sum()


def test_run_kriging_merged_different_lithologies(geo_model, domains, conditioning_data):
    """Merging two different lithologies that should share one property model."""
    lith_array, fault_array, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)

    different_lith_pair = next(
        (a, b) for a in domain_keys for b in domain_keys if a != b and a[0] != b[0]
    )
    group = (different_lith_pair[0], different_lith_pair[1])
    configs = {group: KrigingDomainConfig(model=model)}
    field = run_kriging(geo_model, conditioning_data, configs)

    assert set(field.domain_keys) == set(group)
    expected_mask = group_mask(lith_array, fault_array, group)
    assert np.sum(~np.isnan(field.values)) == expected_mask.sum()


def test_overlapping_groups_raise():
    domain_a = (2, 0)
    domain_b = (2, 1)
    with pytest.raises(ValueError):
        validate_disjoint_groups({domain_a: object(), (domain_a, domain_b): object()}.keys())

    # non-overlapping groups are fine
    validate_disjoint_groups({domain_a: object(), (domain_b,): object()}.keys())


def test_run_simulation_conditioned(geo_model, domains, conditioning_data):
    lith_array, fault_array, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300, nugget=0.1)
    domain_key = domain_keys[0]

    configs = {domain_key: SimulationDomainConfig(model=model, seed=42)}
    field = run_simulation(geo_model, conditioning_data, configs)

    assert field.domain_keys == [domain_key]
    assert np.all(np.isnan(field.variance))
    mask = domain_mask(lith_array, fault_array, domain_key)
    assert np.sum(~np.isnan(field.values)) == mask.sum()


def test_run_simulation_unconditioned(geo_model, domains, conditioning_data):
    _, _, domain_keys = domains
    model = gs.Gaussian(dim=3, var=4, len_scale=300)
    domain_key = domain_keys[0]

    configs = {domain_key: SimulationDomainConfig(model=model, conditioned=False, seed=42)}
    field = run_simulation(geo_model, conditioning_data, configs)

    assert field.domain_keys == [domain_key]
    assert not np.all(np.isnan(field.values))


def test_neighborhood_validates_required_params():
    with pytest.raises(ValueError):
        Neighborhood(mode="n_closest")
    with pytest.raises(ValueError):
        Neighborhood(mode="range")
    Neighborhood(mode="n_closest", n=5)
    Neighborhood(mode="range", radius=100.0)
