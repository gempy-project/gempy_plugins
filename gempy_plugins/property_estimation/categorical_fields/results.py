"""Integer-coded facies ensembles on active geological nodes."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geological_coordinates.embedding import LandmarkGeologicalEmbedding


@dataclass
class FaciesRealizations:
    """Torch integer codes 0..K-1; full-grid inactive cells have code -1.

    Names correspond to code order, not to GemPy structural-element IDs.
    """
    values: Any
    embedding: LandmarkGeologicalEmbedding
    names: tuple
    proportions: tuple
    realization_start: int = 0
    metadata: dict = field(default_factory=dict)

    def to_grid(self, realization=0):
        if not isinstance(realization, (int, np.integer)) or not 0 <= realization < len(self.values):
            raise IndexError("realization is outside this batch")
        values = np.full(np.prod(self.embedding.grid_shape), -1, dtype=np.int64)
        values[self.embedding.node_indices] = self.values[realization].detach().cpu().numpy()
        return values.reshape(self.embedding.grid_shape)

    def to_property_field(self, realization=0):
        """Floating codes with NaN outside the unit for existing volume viewers."""
        from ..kriging import PropertyField
        domains = self.embedding.metadata.get('domain_keys')
        if not domains:
            raise ValueError("PropertyField conversion requires GemPy domain_keys metadata")
        values = self.to_grid(realization).astype(float)
        values[values < 0] = np.nan
        return PropertyField(values, np.full(values.shape, np.nan), [tuple(key) for key in domains])
