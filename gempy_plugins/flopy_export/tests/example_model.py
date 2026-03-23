"""
Create a simple example GemPy model for testing MODFLOW export functionality.
"""
import gempy as gp
import numpy as np


def create_simple_gempy_model():
    """Create a minimal GemPy model for testing."""
    
    # Create model
    geo_model = gp.create_geomodel(
        project_name="test_modflow_model",
        extent=[0, 100, 0, 100, -50, 0],  # x, y, z extent
        resolution=[5, 5, 5]  # nx, ny, nz
    )
    
    # Add surfaces
    gp.add_surface_points(
        geo_model,
        surface_points=np.array([
            [25, 25, -10],
            [75, 75, -10],
        ]),
        surface="Layer1"
    )
    
    gp.add_surface_points(
        geo_model,
        surface_points=np.array([
            [25, 25, -30],
            [75, 75, -30],
        ]),
        surface="Layer2"
    )
    
    # Add orientations
    gp.add_orientations(
        geo_model,
        orientations=np.array([
            [50, 50, -20, 0, 0]  # x, y, z, dip, azimuth
        ]),
        surface="Layer1"
    )
    
    # Compute model
    geo_model.compute_model()
    
    return geo_model


if __name__ == "__main__":
    model = create_simple_gempy_model()
    print(f"Model created with extent: {model.grid.regular_grid.extent}")
    print(f"Resolution: {model.grid.regular_grid.resolution}")