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

from typing import Any, Hashable

import cacholote
import pyproj
import shapely
import xarray as xr

from . import _regrid, _spatial_weighted, _time_weighted


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
    obj: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "mean", **kwargs
    )


def spatial_weighted_std(
    obj: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "std", **kwargs
    )


def spatial_weighted_median(
    obj: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).reduce(
        "quantile", q=0.5, **kwargs
    )


def spatial_weighted_statistics(
    obj: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

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
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).rmse(
        obj2, centralise=False, **kwargs
    )


def spatial_weighted_crmse(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).rmse(
        obj2, centralise=True, **kwargs
    )


def spatial_weighted_corr(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _spatial_weighted.SpatialWeighted(obj1, lon_name, lat_name, weights).corr(
        obj2, **kwargs
    )


def spatial_weighted_errors(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
    weights: xr.DataArray | None = None,
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
    weights: DataArray, optional
        Weights to apply (default is cosine of latitude)

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


def _poly_area(lon_bounds, lat_bounds, geod):  # type: ignore  # TODO: add typing
    if len(lon_bounds) == len(lat_bounds) == 2:
        lon_bounds = sorted(lon_bounds) + sorted(lon_bounds, reverse=True)
        lat_bounds = [lat for lat in lat_bounds for _ in range(2)]
    polygon = shapely.Polygon(zip(lon_bounds, lat_bounds))
    return abs(geod.geometry_area_perimeter(polygon)[0])


@cacholote.cacheable
def _cached_grid_cell_area(
    lon_bounds: dict[str, Any],
    lat_bounds: dict[str, Any],
    bounds_dim: set[str],
    geod: pyproj.Geod,
) -> xr.Dataset:
    area = xr.apply_ufunc(
        _poly_area,
        xr.DataArray.from_dict(lon_bounds),
        xr.DataArray.from_dict(lat_bounds),
        input_core_dims=[tuple(bounds_dim) for _ in range(2)],
        kwargs={"geod": geod},
        vectorize=True,
    )
    area.attrs["standard_name"] = "cell_area"
    cf_area: xr.DataArray = area.cf.add_canonical_attributes()
    return cf_area.to_dataset(name="cell_area")


def grid_cell_area(
    obj: xr.Dataset | xr.DataArray, geod: pyproj.Geod = pyproj.Geod(ellps="WGS84")
) -> xr.DataArray:
    """
    Calculate the area of a cell, in meters^2, on a lat/lon grid.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input object with coordinates
    geod: pyproj.Geod
        Projection (default is WGS84)

    Returns
    -------
    xr.DataArray
        Grid cell area
    """
    bounds = []
    bounds_dim = set()
    for coord in ("longitude", "latitude"):
        if coord not in obj.cf.bounds:
            obj = obj.cf.add_bounds(coord)
        da = obj.cf.get_bounds(coord)
        bounds_dim.update(set(da.dims) - set(obj.cf[coord].dims))
        bounds.append(da.to_dict())
    return _cached_grid_cell_area(bounds[0], bounds[1], bounds_dim, geod)["cell_area"]
