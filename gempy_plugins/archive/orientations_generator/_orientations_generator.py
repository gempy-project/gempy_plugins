import enum
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class NearestSurfacePointsSearcher(enum.Enum):
    KNN = 1
    RADIUS = 2


def select_nearest_surfaces_points(surface_points_xyz: np.ndarray, searchcrit: Optional[int|float] = 3,
                                   search_type: NearestSurfacePointsSearcher = NearestSurfacePointsSearcher.KNN,
                                   filter_less_than: Optional[int] = None) -> np.ndarray:
    match search_type:
        case NearestSurfacePointsSearcher.KNN:
            Tree = NearestNeighbors(n_neighbors=searchcrit)
            Tree.fit(surface_points_xyz)
            neighbours_surfaces = Tree.kneighbors(surface_points_xyz, n_neighbors=searchcrit, return_distance=False)
        case NearestSurfacePointsSearcher.RADIUS:
            Tree = NearestNeighbors(radius=searchcrit)
            Tree.fit(surface_points_xyz)
            neighbours_surfaces = Tree.radius_neighbors(surface_points_xyz, radius=searchcrit, return_distance=False)
        case _:
            raise ValueError(f"Invalid search type: {search_type}")
        
    if filter_less_than is not None:
        neighbours_surfaces = [np.array(neigh) for neigh in neighbours_surfaces if len(neigh) >= filter_less_than]
        
    return neighbours_surfaces