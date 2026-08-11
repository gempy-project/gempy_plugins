GIS Data
--------

Converts vector layers (GeoPandas) and raster DEMs into GemPy surface points and
orientations. Covers the GemPy-relevant slice of what `gemgis
<https://github.com/cgre-aachen/gemgis>`_ originally did, reimplemented against the
current GemPy API rather than depended on, and supersedes the archived ``map2gempy``
plugin for this need.

- ``conversion.py`` -- vector/raster -> ``SurfacePointsTable``/``OrientationsTable``:
  ``extract_xyz``/``surface_points_from_geodataframe`` (draping onto a DEM),
  ``orientations_from_dem``, ``orientations_from_surface_points`` (per-formation plane
  fit), ``orientations_from_strike_lines`` (classical strike-line dip/azimuth), plus
  ``relabel_orientations``/``unique_names`` for datasets whose raw labels don't match
  the target formation names (e.g. several digitized clusters per formation), and
  ``raster_to_xyz``/``nearest_contour_elevation`` helpers.
- ``interpolation.py`` -- ``interpolate_raster`` builds a DEM from elevation-tagged
  contours via RBF interpolation.
- ``alignment.py`` -- ``align_to_minimum_extent`` rotates data to shrink GemPy's
  regular-grid footprint; ``crop_vector_to_bbox``/``crop_points_to_bbox``/
  ``crop_raster_to_bbox``/``crop_surface_points_to_bbox``/``crop_orientations_to_bbox``
  crop vector layers, raw point arrays, rasters, and converted tables to one shared
  ``(minx, miny, maxx, maxy, minz, maxz)`` bounding box (the surface-points/
  orientations crops honor Z; the others only ever use the XY part).
- ``plotting.py`` -- ``plot_vector_layer``/``plot_input_layers``/``plot_raster_2d``
  (matplotlib, 2D map view) and ``plot_raster_3d``/``plot_gis_data_3d`` (pyvista, 3D --
  the latter also previews a crop bounding box against the data before applying it).
- ``io.py`` -- ``surface_points_to_dataframe``/``orientations_to_dataframe`` (pretty-
  printable GemPy-format tables) and ``export_surface_points_csv``/
  ``export_orientations_csv`` built on top of them, writing GemPy's own default CSV
  column layout.

``examples/`` (real, vendored tutorial data, CC BY 4.0, see each
``examples/data_0N/README.rst``):

- ``example01.py`` -- planar dipping layers: DEM from contours, orientations from
  strike lines vs. GemPy's own plane-fit alternative.
- ``example02.py`` -- unconformable faulted layers: a fault plus two unconformity-
  bound series, orientations relabeled from several strike-line clusters per formation.
- ``example03.py`` -- data digitized at an arbitrary rotation: aligning back to the
  model axes, then cropping surface points/orientations/topography to one shared 3D
  bounding box to pull out a sub-model.
