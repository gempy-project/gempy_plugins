"""Convert `SurfacePointsTable`/`OrientationsTable` to GemPy's own CSV convention --
the same column layout `gempy.read_surface_points`/`gempy.read_orientations` expect by
default, with the real formation name written out (not the internal numeric `id`).

`surface_points_to_dataframe`/`orientations_to_dataframe` are useful on their own too --
a quick, readable table (e.g. `print(surface_points_to_dataframe(surface_points))`, or
just evaluated bare in a notebook) is a lot easier to read than `SurfacePointsTable`'s
own repr, which shows the internal `id` rather than the formation name. The CSV
exporters below are this same table, written to disk.
"""
import gempy as gp
from gempy_plugins.optional_dependencies import require_pandas


def surface_points_to_dataframe(surface_points: gp.data.SurfacePointsTable) -> "pandas.DataFrame":
    """Build a `X, Y, Z, formation` table from `surface_points` -- GemPy's default
    `read_surface_points` layout, with `formation` holding the real name (not the
    internal numeric `id`)."""
    pd = require_pandas()
    id_to_name = {id_: name for name, id_ in surface_points.name_id_map.items()}
    xyz = surface_points.xyz
    return pd.DataFrame({
        "X"        : xyz[:, 0],
        "Y"        : xyz[:, 1],
        "Z"        : xyz[:, 2],
        "formation": [id_to_name[id_] for id_ in surface_points.data["id"]],
    })


def orientations_to_dataframe(orientations: gp.data.OrientationsTable) -> "pandas.DataFrame":
    """Build a `X, Y, Z, G_x, G_y, G_z, formation` table from `orientations` -- GemPy's
    default `read_orientations` gradient-vector layout, with `formation` holding the
    real name (not the internal numeric `id`)."""
    pd = require_pandas()
    id_to_name = {id_: name for name, id_ in orientations.name_id_map.items()}
    xyz = orientations.xyz
    grads = orientations.grads
    return pd.DataFrame({
        "X"        : xyz[:, 0],
        "Y"        : xyz[:, 1],
        "Z"        : xyz[:, 2],
        "G_x"      : grads[:, 0],
        "G_y"      : grads[:, 1],
        "G_z"      : grads[:, 2],
        "formation": [id_to_name[id_] for id_ in orientations.data["id"]],
    })


def export_surface_points_csv(surface_points: gp.data.SurfacePointsTable, path: str) -> None:
    """Write `surface_points` to `path` as `X, Y, Z, formation` -- see
    `surface_points_to_dataframe`."""
    surface_points_to_dataframe(surface_points).to_csv(path, index=False)


def export_orientations_csv(orientations: gp.data.OrientationsTable, path: str) -> None:
    """Write `orientations` to `path` as `X, Y, Z, G_x, G_y, G_z, formation` -- see
    `orientations_to_dataframe`."""
    orientations_to_dataframe(orientations).to_csv(path, index=False)
