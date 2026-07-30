"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""

import numpy as np
import pandas as pn
import os

from gempy_plugins.optional_dependencies import require_gdal
import matplotlib.pyplot as plt


class LoadDEMGDAL:
    """Class to include height elevation data (e.g. DEMs) with the geological grid
    """

    def __init__(self, path_dem, grid=None, extent=None, delete_temp=True):
        """
        Args:
            path_dem: path where dem is stored. file format: GDAL raster formats
            if grid: cropped to geomodel extent
        """
        self.gdal = require_gdal()
        self.dem = self.gdal.Open(path_dem)

        if isinstance(self.dem, type(None)):
            raise AttributeError(
                'Raster file could not be opened {}. Check if the filepath is correct. If yes,'
                'check if your file fits the requirements of GDALs raster file formats.'.format(path_dem))

        try:
            self.dem_zval = self.dem.ReadAsArray()
        except AttributeError:
            raise AttributeError('Filepath seems to be wrong.')

        self._get_raster_dimensions()

        if extent is not None:
            self.regular_grid_extent = extent
            self.crop2grid()
        elif grid is not None:
            self.regular_grid_extent = grid.extent
            self.crop2grid()
        else:
            print('pass geo_model to automatically crop the DEM to the grid extent')
        print('depending on the size of the raster, this can take a while...')
        self.convert2xyz()

        if delete_temp is True:
            self.dem = None
            os.remove('topo.xyz')
            if os.path.exists('_cropped_DEM.tif'):
                os.remove('_cropped_DEM.tif')

    def _get_raster_dimensions(self):
        """calculates DEM extent, resolution, and max. z extent (d_z)"""
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        z = self.dem_zval
        if np.any(np.array([xskew, yskew])) != 0:
            print('DEM is not north-oriented.')
        lrx = ulx + (self.dem.RasterXSize * xres)
        lry = uly + (self.dem.RasterYSize * yres)
        self.resolution = np.array([(uly - lry) / (-yres), (lrx - ulx) / xres]).astype(int)
        self.extent = np.array([ulx, lrx, lry, uly]).astype(int)
        self.d_z = np.array([z.min(), z.max()])

    def get_values(self):
        return self.values_3D

    def info(self):
        ulx, xres, xskew, uly, yskew, yres = self.dem.GetGeoTransform()
        print('raster extent:  {}\n raster resolution: {}\n Pixel X size {}, Pixel Y size {}'.format(
            self.extent, self.resolution, xres, yres))
        plt.imshow(self.dem_zval, extent=self.extent)  # plot raster as image
        plt.colorbar()

    def crop2grid(self, delete_temp=True):
        """
        Crops raster to extent of the geomodel grid.
        """
        cornerpoints_geo = self._get_cornerpoints(self.regular_grid_extent)
        cornerpoints_dtm = self._get_cornerpoints(self.extent)

        # self.check()

        if np.any(cornerpoints_geo[:2] - cornerpoints_dtm[:2]) != 0:
            path_dest = '_cropped_DEM.tif'
            new_bounds = (self.regular_grid_extent[[0, 2, 1, 3]])
            self.gdal.Warp(
                path_dest,
                self.dem,
                options=self.gdal.WarpOptions(
                    options=['outputBounds'], outputBounds=new_bounds
                )
            )

            self.dem = self.gdal.Open(path_dest)
            self.dem_zval = self.dem.ReadAsArray()
            self._get_raster_dimensions()

        print('Cropped raster to geo_model.grid.extent.')

    def check(self):
        # TODO make this usable
        test = np.logical_and.reduce((self.regular_grid_extent[0] <= self.extent[0],
                                      self.regular_grid_extent[1] >= self.extent[1],
                                      self.regular_grid_extent[2] <= self.extent[2],
                                      self.regular_grid_extent[3] >= self.extent[3]))
        if test:
            cornerpoints_geo = self._get_cornerpoints(self.regular_grid_extent)
            cornerpoints_dtm = self._get_cornerpoints(self.extent)
            plt.scatter(cornerpoints_geo[:, 0], cornerpoints_geo[:, 1], label='grid extent')
            plt.scatter(cornerpoints_dtm[:, 0], cornerpoints_dtm[:, 1], label='raster extent')
            plt.legend(frameon=True, loc='upper left')
            raise AssertionError('The model extent is too different from the raster extent.')

    def convert2xyz(self, del_temp=True):
        """
        Translates the gdal raster object to a numpy array of xyz coordinates.
        """
        path_dest = 'topo.xyz'
        print('storing converted file...')
        shape = self.dem_zval.shape
        if len(shape) == 3:
            shape = shape[1:]
        self.gdal.Translate(
            path_dest,
            self.dem,
            options=self.gdal.TranslateOptions(
                options=['format'],
                format="XYZ"
            )
        )

        xyz = pn.read_csv(path_dest, header=None, sep=' ').values
        self.values_3D = xyz.reshape((*shape, 3), order='C')

        # This is for 3D going from xyz to ijk
        self.values_3D = self.values_3D.swapaxes(0, 1)
        self.values_3D = np.flip(self.values_3D, 1)

        return self.values_3D

    def _resize(self, resx, resy):
        raise NotImplementedError

    def resample(self, new_xres, new_yres, save_path):
        """
        Decrease the pixel size of the raster.

        Args:
            new_xres (int): desired resolution in x-direction
            new_yres (int): desired resolution in y-direction
            save_path (str): filepath to where the output file should be stored

        Returns: Nothing, it writes a raster file with decreased resolution.

        """
        props = self.dem.GetGeoTransform()
        print('current pixel xsize:', props[1], 'current pixel ysize:', -props[-1])
        options = self.gdal.WarpOptions(options=['tr'], xRes=new_xres, yRes=new_yres)
        newfile = self.gdal.Warp(save_path, self.dem, options=options)
        newprops = newfile.GetGeoTransform()
        print('new pixel xsize:', newprops[1], 'new pixel ysize:', -newprops[-1])
        print('file saved in ' + save_path)

    @staticmethod
    def _get_cornerpoints(extent):
        """Get the coordinates of the bounding box.

        Args:
            extent: np.array([xmin, xmax, ymin, ymax)]

        Returns: np.ndarray with corner coordinates

        """
        upleft = ([extent[0], extent[3]])
        lowleft = ([extent[0], extent[2]])
        upright = ([extent[1], extent[3]])
        lowright = ([extent[1], extent[2]])
        return np.array([upleft, lowleft, upright, lowright])
