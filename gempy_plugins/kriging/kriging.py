"""
This file is part of gempy.

Created on 07.08.2019

@author: Jan von Harten
"""

import warnings
from typing import Optional, Sequence

import gempy as gp
import gempy.core.data
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.transforms import Transform
from gempy.core.data.geo_model import GeoModel
from gempy_engine.core.data.engine_grid import EngineGrid

from gempy_engine.core.data.raw_arrays_solution import RawArraysSolution

from gempy_viewer.modules.plot_2d.visualization_2d import Plot2D
from gempy_viewer.modules.plot_2d.drawer_regular_grid_2d import plot_regular_grid_area

try:
    from scipy.spatial.distance import cdist
    from scipy import linalg
except ImportError:
    warnings.warn('scipy.spatial package is not installed.')

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from copy import deepcopy


class Domain:
    def __init__(self, model_solutions: Solutions, transform: Transform, domain: Optional[Sequence] = None, data=None, set_mean=None):

        # set model from a gempy solution
        # TODO: Check if I actually need all this or if its easier to just get grid and lith of the solution
        self.sol: RawArraysSolution = model_solutions.raw_arrays

        # set kriging surfaces, basically in which lithologies to do all this, default is everything
        # TODO: Maybe also allow to pass a gempy regular grid object
        if domain is None:
            domain = np.unique(self.sol.lith_block)

        # ! This works for now only on the dense grid.
        centers: EngineGrid = model_solutions.octrees_output[0].grid
        self.set_domain(
            domain=domain,
            grid_values=transform.apply_inverse(centers.dense_grid.values)
        )

        # set data, default is None
        # TODO: need to figure out a way to then set mean and variance for the SGS and SK
        if data is None:
            data = None  # why do you do this, data is none already if it is none?
        self.set_data(data)

        # basic statistics of data
        # TODO: allow to set this  for SK ???
        if set_mean is None:
            set_mean = np.mean(data[:, 3])
        self.inp_mean = set_mean

        self.inp_var = np.var(data[:, 3])
        self.inp_std = np.sqrt(self.inp_var)

    def set_domain(self, domain: np.ndarray, grid_values: np.ndarray):
        """
        Method to cut domain by array of surfaces. Simply masking the lith_block with array of input lithologies
        applying mask to grid.
        Args:
            domain (np.array)(x,) = array containing all surfaces of interest from the gempy model that
                                            the operation should be performed in
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        # set domain to variable of class
        self.domain = domain

        # mask by array of input surfaces (by id, can be from different series)

        self.mask = np.isin(self.sol.lith_block, self.domain)

        # Apply mask to lith_block and grid
        self.krig_lith = self.sol.lith_block[self.mask]
        self.krig_grid = grid_values[self.mask]

    def set_data(self, data):
        """
        Method to set input data from csv or numpy array.
        Args:
            data (np.array)(x,4)  = array of input data (conditioning) with [:,0]=x coordinate, [:,1]=y coordinate
                                            [:,2]=z coordinate and [:,3]=value of measured property
        Returns:
            ? Nothing cause of self - is this good practice?
            """
        # set domain to variable of class
        self.data = data

        # create dataframe of input data for calling
        d = {'X': data[:, 0], 'Y': data[:, 1], 'Z': data[:, 2], 'property': data[:, 3]}
        self.data_df = pd.DataFrame(data=d)


class VariogramModel(object):

    # class containing all the variogram functionality

    def __init__(self, theoretical_model=None, range_=1, sill=1, nugget=0):

        if theoretical_model is None:
            theoretical_model = 'exponential'
        self.theoretical_model = theoretical_model

        # default
        self.range_ = range_
        self.sill = sill
        self.nugget = nugget

    def calculate_semivariance(self, d):

        if self.theoretical_model == 'exponential':
            gamma = self.exponential_variogram_model(d)
        elif self.theoretical_model == 'gaussian':
            gamma = self.gaussian_variogram_model(d)
        elif self.theoretical_model == 'spherical':
            gamma = self.spherical_variogram_model(d)
        else:
            print('theoretical varigoram model not understood')
        return gamma

    def calculate_covariance(self, d):

        if self.theoretical_model == 'exponential':
            gamma = self.exponential_covariance_model(d)
        elif self.theoretical_model == 'gaussian':
            gamma = self.gaussian_covariance_model(d)
        elif self.theoretical_model == 'spherical':
            gamma = self.spherical_covariance_model(d)
        else:
            print('theoretical varigoram model not understood')
        return gamma

    # TODO: Add more options
    # seems better now by changing psill in covariance model
    def exponential_variogram_model(self, d):
        '''Exponential variogram model, effective range approximately 3r, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-(np.absolute(d) / (self.range_)))) + self.nugget
        return gamma

    def exponential_covariance_model(self, d):
        '''Exponential covariance model, effective range approximately 3r, valid in R3'''
        psill = self.sill - self.nugget
        cov = psill * (np.exp(-(np.absolute(d) / (self.range_))))
        return cov

    def gaussian_variogram_model(self, d):
        '''Gaussian variogram model, effective range approximately sqrt(3r),
        deprecated due to reverse curvature near orgin, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (1. - np.exp(-d ** 2. / (self.range_) ** 2.)) + self.nugget
        return gamma

    def gaussian_covariance_model(self, d):
        '''Gaussian covariance model, effective range approximately sqrt(3r),
        deprecated due to reverse curvature near orgin, valid in R3'''
        psill = self.sill - self.nugget
        gamma = psill * (np.exp(-d ** 2. / (self.range_) ** 2.))
        return gamma

    def spherical_variogram_model(self, d):
        '''Spherical variogram model, effective range equals range parameter, valid in R3'''
        psill = self.sill - self.nugget
        d = d.astype(float)
        gamma = np.piecewise(d, [d <= self.range_, d > self.range_],
                             [lambda d:
                              psill * ((3. * d) / (2. * self.range_)
                                       - (d ** 3.) / (2. * self.range_ ** 3.)) + self.nugget,
                              lambda d: self.sill])
        return gamma

    def spherical_covariance_model(self, d):
        '''Spherical covariance model, effective range equals range parameter, valid in R3'''
        psill = self.sill - self.nugget
        d = d.astype(float)
        gamma = np.piecewise(d, [d <= self.range_, d > self.range_],
                             [lambda d:
                              psill * (1 - ((3. * d) / (2. * self.range_)
                                            - (d ** 3.) / (2. * self.range_ ** 3.))),
                              lambda d: 0])
        return gamma

    # TODO: Make this better and nicer and everything
    # option for covariance
    # display range, sill, nugget, practical range etc.
    def plot(self, type_='variogram', show_parameters=True):

        if show_parameters == True:
            plt.axhline(self.sill, color='black', lw=1)
            plt.text(self.range_ * 2, self.sill, 'sill', fontsize=12, va='center', ha='center', backgroundcolor='w')
            plt.axvline(self.range_, color='black', lw=1)
            plt.text(self.range_, self.sill / 2, 'range', fontsize=12, va='center', ha='center', backgroundcolor='w')

        if type_ == 'variogram':
            d = np.arange(0, self.range_ * 4, self.range_ / 1000)
            plt.plot(d, self.calculate_semivariance(d), label=self.theoretical_model + " variogram model")
            plt.ylabel('semivariance')
            plt.title('Variogram model')
            plt.legend()

        if type_ == 'covariance':
            d = np.arange(0, self.range_ * 4, self.range_ / 1000)
            plt.plot(d, self.calculate_covariance(d), label=self.theoretical_model + " covariance model")
            plt.ylabel('covariance')
            plt.title('Covariance model')
            plt.legend()

        if type_ == 'both':
            d = np.arange(0, self.range_ * 4, self.range_ / 1000)
            plt.plot(d, self.calculate_semivariance(d), label=self.theoretical_model + " variogram model")
            plt.plot(d, self.calculate_covariance(d), label=self.theoretical_model + " covariance model")
            plt.ylabel('semivariance/covariance')
            plt.title('Models of spatial correlation')
            plt.legend()

        plt.xlabel('lag distance')
        plt.ylim(0 - self.sill / 20, self.sill + self.sill / 20)
        plt.xlim(0, self.range_ * 4)


class KrigingFieldSolution(object):

    def __init__(self, domain, variogram_model, results, field_type):

        self.results_df = results
        self.variogram_model = deepcopy(variogram_model)
        self.domain = deepcopy(domain)
        self.field_type = field_type

    def plot_results(self, geo_data, prop='val', direction='y', result='interpolation', cell_number=0, contour=False,
                     cmap='viridis', alpha=0, legend=False, interpolation='nearest', show_data=True):
        """
        TODO WRITE DOCSTRING
        Args:
            geo_data:
            prop: property that should be plotted - "val", "var" or "both"
            direction: x, y or z
            cell_number:
            contour:
            cmap:
            alpha:
            legend:

        Returns:

        """
        a = np.full_like(self.domain.mask, np.nan, dtype=np.double)  # array like lith_block but with nan if outside domain

        est_vals = self.results_df['estimated value'].values
        est_var = self.results_df['estimation variance'].values

        # set values
        if prop == 'val':
            a[np.where(self.domain.mask == True)] = est_vals
        elif prop == 'var':
            a[np.where(self.domain.mask == True)] = est_var
        elif prop == 'both':
            a[np.where(self.domain.mask == True)] = est_vals
            b = np.full_like(self.domain.mask, np.nan, dtype=np.double)
            b[np.where(self.domain.mask == True)] = est_var
        else:
            print('prop must be val var or both')

        # create plot object
        p = _visualization_2d.PlotSolution(geo_data)
        _a, _b, _c, extent_val, x, y = p._slice(direction, cell_number)[:-2]

        # colors
        cmap = cm.get_cmap(cmap)
        cmap.set_bad(color='w', alpha=alpha)  # define color and alpha for nan values

        # plot
        if prop != 'both':
            if show_data:
                plt.scatter(self.domain.data_df[x].values, self.domain.data_df[y].values, marker='*', s=9, c='k')

            _plot.plot_section(geo_data, direction=direction, cell_number=cell_number)
            if contour == True:
                im = plt.contourf(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                  origin='lower', levels=25,
                                  extent=extent_val, interpolation=interpolation)
                if legend:
                    ax = plt.gca()
                    helpers.add_colorbar(axes=ax, label='prop', cs=im)
            else:
                im = plt.imshow(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                                origin='lower',
                                extent=extent_val, interpolation=interpolation)
                if legend:
                    helpers.add_colorbar(im, label='property value', location='right')

        else:
            f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            ax[0].title.set_text('Estimated value')
            im1 = ax[0].imshow(a.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                               origin='lower', interpolation=interpolation,
                               extent=self.domain.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im1, label='property value')
            ax[1].title.set_text('Variance')
            im2 = ax[1].imshow(b.reshape(self.domain.sol.grid.regular_grid.resolution)[_a, _b, _c].T, cmap=cmap,
                               origin='lower', interpolation=interpolation,
                               extent=self.domain.sol.grid.regular_grid.extent[[0, 1, 4, 5]])
            helpers.add_colorbar(im2, label='variance[]')
            plt.tight_layout()


def plot_kriging_results(geo_data: GeoModel, kriging_solution: KrigingFieldSolution, plot_2d: Plot2D,
                         title: str, result_column: list[str]):
    for e, ax in enumerate(plot_2d.axes):
        a = np.full_like(kriging_solution.domain.mask, np.nan, dtype=np.double)
        vals = kriging_solution.results_df[result_column[e]].values
        a[np.where(kriging_solution.domain.mask == True)] = vals

        im = plot_regular_grid_area(
            ax=ax,
            slicer_data=plot_2d.section_data_list[0].slicer_data,
            block=a,
            resolution=geo_data.grid.regular_grid.resolution,
            cmap='viridis',
            norm=None,
        )

        ax.set_title(title)
        plot_2d.fig.colorbar(im, label='Property value')
    plot_2d.fig.show()


# TODO: check with new ordianry kriging and nugget effect
def simple_kriging(a, b, prop, var_mod, inp_mean):
    '''
    Method for simple kriging calculation.
    Args:
        a (np.array): distance matrix containing all distances between target point and moving neighbourhood
        b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop (np.array): array containing scalar property values of locations in moving neighbourhood
        var_mod: variogram model object
    Returns:
        result (float?): single scalar property value estimated for target location
        std_ok (float?): single scalar variance value for estimate at target location
    '''

    # empty matrix building
    shape = len(a)
    C = np.zeros((shape, shape))
    c = np.zeros((shape))
    w = np.zeros((shape))

    # Filling matrices with covariances based on calculated distances
    C[:shape, :shape] = var_mod.calculate_covariance(b)  # ? cov or semiv
    c[:shape] = var_mod.calculate_covariance(a)  # ? cov or semiv

    # nugget effect for simple kriging - dont remember why i set this actively, should be the same
    # np.fill_diagonal(C, self.sill)

    # TODO: find way to check quality of matrix and solutions for instability
    # Solve Kriging equations
    w = np.linalg.solve(C, c)

    # calculating estimate and variance for kriging
    pred_var = var_mod.sill - np.sum(w * c)
    # Note that here the input mean is required, if kriged mean equivalent to OK
    result = inp_mean + np.sum(w * (prop - inp_mean))

    return result, pred_var


def ordinary_kriging(a, b, prop, var_mod):
    '''
    Method for ordinary kriging calculation.
    Args:
        a (np.array): distance matrix containing all distances between target point and moving neighbourhood
        b (np.array): distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop (np.array): array containing scalar property values of locations in moving neighbourhood
        var_mod: variogram model object
    Returns:
        result (float?): single scalar property value estimated for target location
        std_ok (float?): single scalar variance value for estimate at target location
    '''

    # empty matrix building for OK
    shape = len(a)
    C = np.zeros((shape + 1, shape + 1))
    c = np.zeros((shape + 1))
    w = np.zeros((shape + 1))

    # filling matirces based on model for spatial correlation
    C[:shape, :shape] = var_mod.calculate_semivariance(b)
    c[:shape] = var_mod.calculate_semivariance(a)

    # matrix setup - compare pykrige, special for OK
    np.fill_diagonal(C, 0)  # this needs to be done as semivariance for distance 0 is 0 by definition
    C[shape, :] = 1.0
    C[:, shape] = 1.0
    C[shape, shape] = 0.0
    c[shape] = 1.0

    # This is if we want exact interpolator
    # but be aware that it strictly forces estimates to go through data points
    # c[c == self.nugget] = 0

    # TODO: find way to check quality of matrix and solutions for instability
    # Solve Kriging equations
    w = np.linalg.solve(C, c)

    # calculating estimate and variance for kriging
    pred_var = w[shape] + np.sum(w[:shape] * c[:shape])
    result = np.sum(w[:shape] * prop)

    return result, pred_var


def _ordinary_kriging_from_lu(a, prop, var_mod, LU, piv):
    shape = len(a)
    c = np.zeros(shape + 1)
    c[:shape] = var_mod.calculate_semivariance(a)
    c[shape] = 1.0
    w = linalg.lu_solve((LU, piv), c)
    pred_var = w[shape] + np.sum(w[:shape] * c[:shape])
    result = np.sum(w[:shape] * prop)
    return result, pred_var


def _simple_kriging_from_lu(a, prop, var_mod, inp_mean, LU, piv):
    shape = len(a)
    c = np.zeros(shape)
    c[:shape] = var_mod.calculate_covariance(a)
    w = linalg.lu_solve((LU, piv), c)
    pred_var = var_mod.sill - np.sum(w * c)
    result = inp_mean + np.sum(w * (prop - inp_mean))
    return result, pred_var


def create_kriged_field(domain, variogram_model, distance_type='euclidian',
                        moving_neighbourhood='all', kriging_type='OK', n_closest_points=20):
    '''
    Method to create a kriged field over the defined grid of the gempy solution depending on the defined
    input data (conditioning).
    Returns:
        self.results_df (pandas dataframe):   Dataframe containing coordinates, kriging estimate
                                                    and kriging variance for each grid point
    '''
    n_grid = len(domain.krig_grid)
    n_data = len(domain.data[:, 3])
    kriging_result_vals = np.zeros(n_grid)
    kriging_result_vars = np.zeros(n_grid)

    if distance_type == 'euclidian':
        dist_grid_to_all = cdist(domain.krig_grid, domain.data[:, :3])

    if moving_neighbourhood == 'all' and n_data > 0:
        dist_all_to_all = cdist(domain.data[:, :3], domain.data[:, :3])

        if kriging_type == 'OK':
            shape = n_data
            C = np.zeros((shape + 1, shape + 1))
            C[:shape, :shape] = variogram_model.calculate_semivariance(dist_all_to_all)
            np.fill_diagonal(C, 0)
            C[shape, :] = 1.0
            C[:, shape] = 1.0
            C[shape, shape] = 0.0
            LU, piv = linalg.lu_factor(C)
            prop = domain.data[:, 3]
            for i in range(n_grid):
                val, var = _ordinary_kriging_from_lu(dist_grid_to_all[i], prop, variogram_model, LU, piv)
                kriging_result_vals[i] = val
                kriging_result_vars[i] = var

        elif kriging_type == 'SK':
            shape = n_data
            C = np.zeros((shape, shape))
            C[:shape, :shape] = variogram_model.calculate_covariance(dist_all_to_all)
            LU, piv = linalg.lu_factor(C)
            prop = domain.data[:, 3]
            for i in range(n_grid):
                val, var = _simple_kriging_from_lu(dist_grid_to_all[i], prop, variogram_model, domain.inp_mean, LU, piv)
                kriging_result_vals[i] = val
                kriging_result_vars[i] = var
        else:
            print("FATAL ERROR: Kriging type not understood")
    else:
        if moving_neighbourhood != 'all':
            dist_all_to_all = cdist(domain.data[:, :3], domain.data[:, :3])

        for i in range(n_grid):
            if moving_neighbourhood == 'all':
                a = dist_grid_to_all[i]
                b = dist_all_to_all
                prop = domain.data[:, 3]

            elif moving_neighbourhood == 'n_closest':
                n_neighbors = min(n_closest_points, n_data)
                nearest_idx = np.argpartition(dist_grid_to_all[i], n_neighbors - 1)[:n_neighbors]
                nearest_idx = nearest_idx[np.argsort(dist_grid_to_all[i][nearest_idx])]
                a = dist_grid_to_all[i][nearest_idx]
                prop = domain.data[:, 3][nearest_idx]
                b = dist_all_to_all[np.ix_(nearest_idx, nearest_idx)]

            elif moving_neighbourhood == 'range':
                aux = np.where(dist_grid_to_all[i] <= variogram_model.range_)[0]
                if len(aux) == 0:
                    aux = np.array([0])
                a = dist_grid_to_all[i][aux]
                prop = domain.data[:, 3][aux]
                b = dist_all_to_all[np.ix_(aux, aux)]

            else:
                print("FATAL ERROR: Moving neighbourhood not understood")
                continue

            if kriging_type == 'OK':
                val, var = ordinary_kriging(a, b, prop, variogram_model)
            elif kriging_type == 'SK':
                val, var = simple_kriging(a, b, prop, variogram_model, domain.inp_mean)
            elif kriging_type == 'UK':
                print("Universal Kriging not implemented")
                continue
            else:
                print("FATAL ERROR: Kriging type not understood")
                continue

            kriging_result_vals[i] = val
            kriging_result_vars[i] = var

    d = {'X': domain.krig_grid[:, 0], 'Y': domain.krig_grid[:, 1], 'Z': domain.krig_grid[:, 2],
         'estimated value': kriging_result_vals, 'estimation variance': kriging_result_vars}

    results_df = pd.DataFrame(data=d)

    return KrigingFieldSolution(domain, variogram_model, results_df, field_type='interpolation')


def create_gaussian_field(domain, variogram_model, distance_type='euclidian',
                          moving_neighbourhood='n_closest', kriging_type='OK', n_closest_points=20):
    '''
    Method to perform Sequential Gaussian Simulation over the defined grid of the gempy
    solution depending on the defined input data (conditioning).

    Defaults to a local neighbourhood (n_closest) to avoid building a full
    N x N distance matrix for dense grids.

    Returns:
        KrigingFieldSolution with results_df containing coordinates, estimate and variance.
    '''
    n_grid = len(domain.krig_grid)
    n_data = len(domain.data[:, 3])

    if moving_neighbourhood == 'all' and n_grid > 1000:
        raise ValueError(
            f"moving_neighbourhood='all' requires a ({n_grid+n_data}) x ({n_grid+n_data}) "
            f"distance matrix which is infeasible for {n_grid} grid nodes. "
            f"Use moving_neighbourhood='n_closest' or 'range' instead."
        )

    shuffled_grid = domain.krig_grid.copy()
    np.random.shuffle(shuffled_grid)

    sgs_locations = np.vstack((domain.data[:, :3], shuffled_grid))
    sgs_prop_updating = domain.data[:, 3].copy()

    estimation_var = np.zeros(n_grid)

    if moving_neighbourhood == 'all':
        dist_all_to_all = cdist(sgs_locations, sgs_locations)

    active_data = n_data

    for i in range(n_grid):
        target_coords = sgs_locations[active_data:active_data + 1]

        if moving_neighbourhood == 'all':
            a = dist_all_to_all[active_data, :active_data]
            b = dist_all_to_all[:active_data, :active_data]
            prop = sgs_prop_updating[:active_data]
        else:
            target_dists = cdist(target_coords, sgs_locations[:active_data, :3])[0]

            if moving_neighbourhood == 'n_closest':
                n_neighbors = min(n_closest_points, active_data)
                if n_neighbors == 0:
                    val = domain.inp_mean if kriging_type == 'SK' else np.mean(domain.data[:, 3])
                    var = variogram_model.sill
                    std_ = np.sqrt(max(var, 0))
                    estimate = np.random.normal(val, scale=std_)
                    sgs_prop_updating = np.append(sgs_prop_updating, estimate)
                    estimation_var[i] = var
                    active_data += 1
                    continue
                nearest_idx = np.argpartition(target_dists, n_neighbors - 1)[:n_neighbors]
                nearest_idx = nearest_idx[np.argsort(target_dists[nearest_idx])]
                a = target_dists[nearest_idx]
                prop = sgs_prop_updating[nearest_idx]
                b = cdist(sgs_locations[nearest_idx, :3], sgs_locations[nearest_idx, :3])

            elif moving_neighbourhood == 'range':
                aux = np.where(target_dists <= variogram_model.range_)[0]
                if len(aux) == 0:
                    aux = np.array([0])
                a = target_dists[aux]
                prop = sgs_prop_updating[aux]
                b = cdist(sgs_locations[aux, :3], sgs_locations[aux, :3])

            else:
                print("FATAL ERROR: Moving neighbourhood not understood")
                continue

        if kriging_type == 'OK':
            val, var = ordinary_kriging(a, b, prop, variogram_model)
        elif kriging_type == 'SK':
            val, var = simple_kriging(a, b, prop, variogram_model, domain.inp_mean)
        elif kriging_type == 'UK':
            print("Universal Kriging not implemented")
        else:
            print("FATAL ERROR: Kriging type not understood")
            continue

        std_ = np.sqrt(max(var, 0.0))
        estimate = np.random.normal(val, scale=std_)

        sgs_prop_updating = np.append(sgs_prop_updating, estimate)
        estimation_var[i] = var
        active_data += 1

    simulated_prop = sgs_prop_updating[n_data:]

    d = {'X': shuffled_grid[:, 0], 'Y': shuffled_grid[:, 1], 'Z': shuffled_grid[:, 2],
         'estimated value': simulated_prop, 'estimation variance': estimation_var}

    results_df = pd.DataFrame(data=d)
    results_df = results_df.sort_values(['X', 'Y', 'Z'])

    return KrigingFieldSolution(domain, variogram_model, results_df, field_type='simulation')
