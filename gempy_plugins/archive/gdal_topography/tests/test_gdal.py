import subsurface as ss


def test_real_grid_ales():
    struct: ss.StructuredData = ss.reader.read_structured_topography(
        path="./data/_cropped_DEM_coarse.tif",
        crop_to_extent=[729550.0, 751500.0, 1913500.0, 1923650.0]
    )
    print(struct)
    