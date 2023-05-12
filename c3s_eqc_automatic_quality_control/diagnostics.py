"""C3S EQC Automatic Quality Control.

This module gathers available diagnostics.
"""

# Copyright 2022, European Union.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Hashable
from typing import Any

import pyproj
import xarray as xr
from xarray.core.common import DataWithCoords

from . import _grid_cell_area, _regrid, _spatial_weighted, _time_weighted


def regrid(
    obj: xr.DataArray | xr.Dataset,
    grid_out: xr.DataArray | xr.Dataset,
    method: str,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Regrid object.

    Parameters
    ----------
    obj: DataArray or Dataset
        Object to regrid
    grid_out: DataArray or Dataset
        Output grid
    method: str
        xesmf interpolation method
    **kwargs: Any
        keyword arguments for xesmf

    Returns
    -------
    DataArray or Dataset
        Interpolated object
    """
    regridder = _regrid.cached_regridder(obj, grid_out, method, **kwargs)
    obj = regridder(obj, keep_attrs=True)
    return obj


def time_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate time weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: hashable, optional
        Name of time coordinate
    weights: DataArray, optional
        Weights to apply (default is days per month)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", None, **kwargs
    )


def seasonal_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate seasonal weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: hashable, optional
        Name of time coordinate
    weights: DataArray, optional
        Weights to apply (default is days per month)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", "season", **kwargs
    )


def annual_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate annual weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: hashable, optional
        Name of time coordinate
    weights: DataArray, optional
        Weights to apply (default is days per month)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", "year", **kwargs
    )


def spatial_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    lon_name, lat_name: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "mean", **kwargs
    )


def spatial_weighted_std(
    obj: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial std of ds with latitude weighting.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    lon_name, lat_name: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "std", **kwargs
    )


def spatial_weighted_median(
    obj: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    lon_name, lat_name: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "quantile", q=0.5, **kwargs
    )


def spatial_weighted_statistics(
    obj: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean, std, and median of ds with latitude weighting.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    lon_name, lat_name: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    sw = _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights)
    objects = []
    for func in ("mean", "std", "median"):
        if func == "median":
            obj = sw.reduce("quantile", q=0.5, **kwargs)
        else:
            obj = sw.reduce(func, **kwargs)
        objects.append(obj.expand_dims(diagnostic=[func]))
    ds = xr.merge(objects)
    return ds[obj.name] if isinstance(obj, xr.DataArray) else ds


def spatial_weighted_rmse(
    obj1: xr.DataArray | xr.Dataset,
    obj2: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial rmse with latitude weighting.

    Parameters
    ----------
    obj1, obj2: DataArray or Dataset
        Input data
    lon, lat: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).rmse(
        obj2, centralise=False, **kwargs
    )


def spatial_weighted_crmse(
    obj1: xr.DataArray | xr.Dataset,
    obj2: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial crmse with latitude weighting.

    Parameters
    ----------
    obj1, obj2: DataArray or Dataset
        Input data
    lon, lat: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).rmse(
        obj2, centralise=True, **kwargs
    )


def spatial_weighted_corr(
    obj1: xr.DataArray | xr.Dataset,
    obj2: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial correlation with latitude weighting.

    Parameters
    ----------
    obj1, obj2: xr.Dataset or xr.DataArray
        Input data
    lon, lat: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).corr(
        obj2, **kwargs
    )


def spatial_weighted_errors(
    obj1: xr.DataArray | xr.Dataset,
    obj2: xr.DataArray | xr.Dataset,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate rmse, crmse, and correlation with latitude weighting.

    Parameters
    ----------
    obj1, obj2: xr.Dataset or xr.DataArray
        Input data
    lon, lat: str, optional
        Name of longitude/latitude coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the cosine of the latitude
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    sw = _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights)
    objects = []
    for func in ("rmse", "crmse", "corr"):
        if func.endswith("rmse"):
            centralise = True if func.startswith("c") else False
            obj = sw.rmse(obj2, centralise=centralise, **kwargs)
        else:
            obj = getattr(sw, func)(obj2, **kwargs)
        objects.append(obj.expand_dims(diagnostic=[func]))
    ds = xr.merge(objects)
    return ds[obj1.name] if isinstance(obj1, xr.DataArray) else ds


def grid_cell_area(
    obj: DataWithCoords, geod: pyproj.Geod = pyproj.Geod(ellps="WGS84")
) -> xr.DataArray:
    """
    Calculate the area of a cell, in meters^2, on a lat/lon grid.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input object with coordinates
    geod: pyproj.Geod
        Projection (default is WGS84)

    Returns
    -------
    DataArray
        Grid cell area
    """
    ds = obj if isinstance(obj, xr.Dataset) else obj.to_dataset(name=obj.name or "None")
    ds = ds.cf.add_bounds({"longitude", "latitude"} - set(ds.cf.bounds))
    return _grid_cell_area.cached_grid_cell_area(
        ds.cf.get_bounds("longitude").to_dict(),
        ds.cf.get_bounds("latitude").to_dict(),
        geod,
    )["cell_area"]
