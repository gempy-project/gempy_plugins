"""
Populating a Structural Model with Properties
================================================

Domain-aware kriging and simulation with GSTools, respecting a model's lithology and
fault-block boundaries.

Based on https://github.com/cgre-aachen/egu25_gempy_workshop/blob/main/outlook_notebooks/Property_modeling_EGU25.ipynb
"""

# %%
# gempy builds structural models out of boundary surfaces between rock units, but often
# what you actually care about is a property *within* those units -- porosity, ore
# grade, whatever you're measuring. This plugin populates a computed model's regular
# grid with such a property via kriging or simulation, using `GSTools
# <https://geostat-framework.readthedocs.io/projects/gstools/en/stable/>`_ for all the
# actual geostatistics -- variogram models, kriging, and simulation are all plain
# GSTools objects, passed straight through with no wrapping.
#
# The one thing this plugin adds on top of GSTools itself is domain awareness: a
# "domain" here is a (lithology, fault-block) pair, so property estimation can respect
# both kinds of structural boundary -- including the same lithology occurring on both
# sides of a fault, which needs to stay two separate domains, not one.

# %%
import numpy as np

import gempy as gp
import gstools as gs
from gempy_plugins.property_estimation.conditioning_data import ConditioningData
from gempy_plugins.property_estimation.domains import compute_domains, describe_domains
from gempy_plugins.property_estimation.kriging import KrigingDomainConfig, run_kriging
from gempy_plugins.property_estimation.plotting import (
    plot_conditioning_data, plot_domains, plot_fault_blocks, plot_property_field,
)

np.random.seed(1)

# %%
# A faulted structural model
# -----------------------------
# The same model used in the reference notebook above: three lithologies offset by a
# fault -- except here ``rock3`` is placed *younger* than the fault (first in the
# mapping below), so the fault doesn't actually offset it. That's deliberate: it's the
# motivating example for domain merging further down.

# %%
data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'

geo_model = gp.create_geomodel(
    project_name='combination',
    extent=[0, 2500, 0, 1000, 0, 1000],
    resolution=[50, 20, 20],
    importer_helper=gp.data.ImporterHelper(
        path_to_orientations=data_path + "/data/input_data/jan_models/model7_orientations.csv",
        path_to_surface_points=data_path + "/data/input_data/jan_models/model7_surface_points.csv"
    )
)
gp.map_stack_to_surfaces(
    gempy_model=geo_model,
    mapping_object={
        "Strat_Series1": ('rock3'),
        "Fault_Series": ('fault'),
        "Strat_Series2": ('rock2', 'rock1'),
    }
)
gp.set_is_fault(geo_model, ["Fault_Series"])
gp.compute_model(geo_model)

# %%
# Computing and inspecting the domains
# ---------------------------------------
# Before doing anything else, it's worth seeing exactly how the model got split into
# domains -- `compute_domains` combines lithology and fault-block membership, and
# `plot_domains` visualizes the result directly:

# %%
lith_array, fault_array, domain_keys = compute_domains(geo_model)
describe_domains(geo_model, domain_keys)

# %%
# Each lithology gets its own base color, with its fault-block variants shown as
# shades of that color -- so "same rock unit, different fault block" is obvious:

# %%
plot_domains(geo_model, lith_array, fault_array, domain_keys)

# %%
# Shades alone don't say *which* fault block is which, though -- and that's exactly
# what you need to decide whether domains should be merged. ``plot_fault_blocks``
# shows fault blocks alone:

# %%
plot_fault_blocks(geo_model, fault_array)

# %%
# Why some domains need merging
# ---------------------------------
# ``rock3`` was placed younger than the fault above, so it shouldn't be offset by it
# at all -- and ``fault_relations`` confirms that:

# %%
geo_model.structural_frame.fault_relations

# %%
# And yet ``rock3`` still shows up as *two* separate domains in the plots above, one
# per fault block. That's because fault-block membership is purely geometric -- which
# side of the fault's surface a cell falls on -- independent of whether
# ``fault_relations`` actually applies an offset to a given lithology. The same thing
# happens with a real fault of negligible offset: geometrically there are still two
# blocks, even though nothing actually moved. Either way, the fix is the same: merge
# the domains that should really be treated as one. This is a purely structural
# decision -- it doesn't need any conditioning data yet, just the domain keys
# themselves. A ``domain_configs`` key can be a tuple of domain keys instead of a
# single one, to merge them later:

# %%
rock3_merged = (domain_keys[0], domain_keys[1])

# %%
# Conditioning data
# --------------------
# Standing in for real property samples (e.g. from boreholes): scattered points with a
# measured value, assigned to their nearest domain.

# %%
n_samples = 300
conditioning_data = ConditioningData(
    x=np.random.uniform(0, 2500, n_samples),
    y=np.random.uniform(0, 1000, n_samples),
    z=np.random.uniform(0, 1000, n_samples),
    values=np.random.normal(15, 3, n_samples),
)
conditioning_data.assign_domains(geo_model, lith_array, fault_array)

# %%
# It's worth seeing where these samples actually sit relative to the structure before
# running anything -- ``plot_conditioning_data`` overlays them (colored by value) on
# the model's surfaces alone:

# %%
plot_conditioning_data(geo_model, conditioning_data)

# %%
# Kriging, with different settings per domain
# -----------------------------------------------
# A domain is processed only if it has an entry in ``domain_configs`` -- so kriging can
# be run over a chosen subset of domains, each with its own variogram model and kriging
# method. To make that concrete, three domains below each get different treatment: the
# merged ``rock3`` domain and ``basement`` share a Gaussian model, ``rock2`` gets an
# Exponential model instead, and kriging type varies independently of that -- Ordinary
# for ``rock3``/``rock2``, Simple (with a fixed mean) for ``basement``. A small nugget
# is added to the Gaussian model since a smooth, nugget-free Gaussian covariance model
# can become numerically ill-conditioned once a domain has many conditioning points --
# worth knowing about if you see wildly out-of-range kriged values. The Exponential
# model doesn't have that problem even without a nugget.

# %%
gaussian_model = gs.Gaussian(dim=3, var=4, len_scale=400, nugget=0.1)
exponential_model = gs.Exponential(dim=3, var=4, len_scale=400)

domain_configs = {
    rock3_merged: KrigingDomainConfig(model=gaussian_model),
    domain_keys[2]: KrigingDomainConfig(model=exponential_model),
    domain_keys[6]: KrigingDomainConfig(
        model=gaussian_model,
        krige_class=gs.krige.Simple,
        krige_kwargs={'mean': 15},
    ),
}
field = run_kriging(geo_model, conditioning_data, domain_configs)

# %%
# Only the three configured domains are populated -- everything else stays `nan` and
# simply isn't rendered:

# %%
plot_property_field(geo_model, field)
