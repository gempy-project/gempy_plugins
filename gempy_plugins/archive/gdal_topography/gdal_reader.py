from gdal_topography.load_dem_gdal import LoadDEMGDAL


def load_from_gdal(self, filepath):
    dem = LoadDEMGDAL(filepath, extent=self.extent)
    self._x, self._y = None, None
    self.set_values(dem.get_values())
