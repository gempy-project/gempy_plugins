import os

import numpy as np
import pytest

import gempy as gp
from gempy_plugins.topology_analysis import topology as tp

data_path = os.path.dirname(__file__) + '/../examples/data'


@pytest.fixture(scope='module')
def geo_model():
    geo_model = gp.create_geomodel(
        project_name='Model_Tutorial6',
        extent=[0, 3000, 0, 20, 0, 2000],
        resolution=[50, 10, 67],
        refinement=1,
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=data_path + "/ch6_data_fol.csv",
            path_to_surface_points=data_path + "/ch6_data_interf.csv",
        )
    )
    gp.map_stack_to_surfaces(
        gempy_model=geo_model,
        mapping_object={
            "fault": "Fault",
            "Rest": ('Layer 2', 'Layer 3', 'Layer 4', 'Layer 5')
        }
    )
    gp.set_is_fault(geo_model, ['fault'])
    geo_model.interpolation_options.evaluation_options.mesh_extraction = False
    gp.compute_model(geo_model)
    return geo_model


@pytest.fixture(scope='module')
def topology(geo_model):
    return tp.compute_topology(geo_model)


def test_compute_topology(topology):
    edges, centroids = topology
    assert len(edges) > 0
    assert len(centroids) > 0
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)


def test_lith_and_fault_lookup_tables_are_consistent(geo_model, topology):
    edges, centroids = topology
    lith_lot = tp.get_lot_node_to_lith_id(geo_model, centroids)
    lith_to_node = tp.get_lot_lith_to_node_id(lith_lot)
    fault_lot = tp.get_lot_node_to_fault_block(geo_model, centroids)

    # every node listed for a given lith id must map back to that same lith id
    for lith_id, node_ids in lith_to_node.items():
        for node_id in node_ids:
            assert lith_lot[node_id] == lith_id

    # the model has one fault, so nodes should split into exactly two fault blocks
    assert set(fault_lot.values()) == {0, 1}


def test_adjacency_matrix_is_square_and_symmetric(geo_model, topology):
    edges, centroids = topology
    M = tp.get_adjacency_matrix(geo_model, edges, centroids)
    assert M.shape[0] == M.shape[1]
    assert M.dtype == bool
    assert np.array_equal(M, M.T)


def test_check_adjacency_and_get_adjacencies(geo_model, topology):
    edges, centroids = topology
    dedges, dcentroids = tp.get_detailed_labels(geo_model, edges, centroids)

    adjacent = tp.get_adjacencies(dedges, "5_1")
    assert len(adjacent) > 0

    for neighbour in adjacent:
        assert tp.check_adjacency(dedges, "5_1", neighbour)

    # a node is never adjacent to a label that isn't in the graph at all
    assert not tp.check_adjacency(dedges, "5_1", "unrelated_label")


def test_jaccard_index():
    edges1 = {(0, 1), (1, 2), (2, 3)}
    edges2 = {(0, 1), (1, 2), (3, 4)}
    # intersection = {(0,1),(1,2)} -> 2, union = {(0,1),(1,2),(2,3),(3,4)} -> 4
    assert tp.jaccard_index(edges1, edges2) == 0.5
    assert tp.jaccard_index(edges1, edges1) == 1.0


def test_count_unique_topologies():
    topologies = [
        {(0, 1), (1, 2)},
        {(0, 1), (1, 2)},
        {(0, 2), (2, 3)},
    ]
    unique_edges, counts, idx = tp.count_unique_topologies(topologies)
    assert len(unique_edges) == 2
    assert sorted(counts.tolist()) == [1, 2]