import gempy as gp
import os

from orientations_generator import select_nearest_surfaces_points, NearestSurfacePointsSearcher

input_path = os.path.dirname(__file__) + '/../../examples/data'
data_path = os.path.abspath('../../../tests/data')


def _model_factory():
    # Importing the data from CSV-files and setting extent and resolution
    geo_data = gp.create_geomodel(
        extent=[0, 2000, 0, 2000, 0, 2000],
        resolution=[50, 50, 50],
        importer_helper=gp.data.ImporterHelper(
            path_to_orientations=f"{data_path}/model5_orientations.csv",
            path_to_surface_points=f"{data_path}/model5_surface_points.csv",
        )
    )
    return geo_data


def test_set_orientations():
    geo_data = _model_factory()

    orientations: gp.data.OrientationsTable = gp.create_orientations_from_surface_points_coords(
        xyz_coords=geo_data.surface_points.xyz
    )

    gp.add_orientations(
        geo_model=geo_data,
        x=orientations.data['X'],
        y=orientations.data['Y'],
        z=orientations.data['Z'],
        pole_vector=orientations.grads,
        elements_names=geo_data.structural_frame.elements_names[0],
    )


def test_select_nearest_surface_points():

    geo_model = _model_factory()
    print(geo_model)
    
    element: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("fault")

    # find neighbours
    knn = select_nearest_surfaces_points(
        surface_points_xyz=element.surface_points.xyz,
        searchcrit=3
    )

    radius = select_nearest_surfaces_points(
        surface_points_xyz=element.surface_points.xyz,
        searchcrit=200.,
        search_type=NearestSurfacePointsSearcher.RADIUS
    )
    
    return knn


def test_set_orientation_from_neighbours():
    geo_model = _model_factory()

    print(geo_model)

    element: gp.data.StructuralElement = geo_model.structural_frame.get_element_by_name("fault")

    # find neighbours
    knn = select_nearest_surfaces_points(
        surface_points_xyz=element.surface_points.xyz,
        searchcrit=3
    )

    orientations: gp.data.OrientationsTable = gp.create_orientations_from_surface_points_coords(
        xyz_coords=geo_model.surface_points.xyz,
        subset=knn
    )       

    gp.add_orientations(
        geo_model=geo_model,
        x=orientations.data['X'],
        y=orientations.data['Y'],
        z=orientations.data['Z'],
        pole_vector=orientations.grads,
        elements_names=geo_model.structural_frame.elements_names[0],
    )
    
    return knn
