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

from typing import Any

import cacholote
import numpy as np
import xarray as xr
import xesmf as xe


def _get_lon_and_lat(
    obj: xr.Dataset | xr.DataArray, lon: str | None, lat: str | None
) -> tuple[str, str]:
    if lon is None:
        (lon,) = obj.cf.coordinates["longitude"]
    if lat is None:
        (lat,) = obj.cf.coordinates["latitude"]
    return lon, lat


def _get_time(obj: xr.Dataset | xr.DataArray, time: str | None) -> str:
    if time is None:
        (time,) = obj.cf.coordinates["time"]
    return time


def _spatial_weights(
    obj: xr.Dataset | xr.DataArray, lon: str | None = None, lat: str | None = None
) -> xr.DataArray:
    lon, lat = _get_lon_and_lat(obj, lon, lat)
    cos = np.cos(np.deg2rad(obj[lat]))
    weights: xr.DataArray = cos / (cos.sum(lat) * len(obj[lon]))
    return weights


@cacholote.cacheable
def _regridder_weights(
    dict_in: dict[str, Any], dict_out: dict[str, Any], method: str, **kwargs: Any
) -> xr.Dataset:
    weights: xr.Dataset = xe.Regridder(
        xr.Dataset.from_dict(dict_in), xr.Dataset.from_dict(dict_out), method, **kwargs
    ).weights
    return weights


def _grid_to_dict(grid: xr.Dataset) -> dict[str, Any]:
    coords = grid.cf.coordinates["longitude"] + grid.cf.coordinates["latitude"]
    grid_dict: dict[str, Any] = grid[coords].to_dict()
    grid_dict.pop("attrs")
    return grid_dict


def _regridder(
    grid_in: xr.Dataset, grid_out: xr.Dataset, method: str, **kwargs: Any
) -> xe.Regridder:
    # Remove metadata and cache using dicts
    dict_in = _grid_to_dict(grid_in)
    dict_out = _grid_to_dict(grid_out)
    kwargs["weights"] = _regridder_weights(dict_in, dict_out, method, **kwargs)
    return xe.Regridder(grid_in, grid_out, method, **kwargs)


def regrid(
    obj: xr.Dataset, grid_out: xr.Dataset, method: str, **kwargs: Any
) -> xr.Dataset:
    regridder = _regridder(obj, grid_out, method, **kwargs)
    obj = regridder(obj, keep_attrs=True)
    return obj


def seasonal_weighted_mean(obj: xr.Dataset, time: str | None = None) -> xr.Dataset:
    """
    Calculate seasonal weighted mean.

    Parameters
    ----------
    obj: xr.Dataset
        Input data
    time: str, optional
        Name of time coordinate

    Returns
    -------
    reduced object
    """
    time = _get_time(obj, time)

    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        obj = obj.convert_calendar("noleap", align_on="date")
        month_length = obj[time].dt.days_in_month
        weights = (
            month_length.groupby(f"{time}.season")
            / month_length.groupby(f"{time}.season").sum()
        )
        obj = (obj * weights).groupby(f"{time}.season").sum(dim=time)
    return obj


def annual_weighted_mean(obj: xr.Dataset, time: str | None = None) -> xr.Dataset:
    """
    Calculate annual weighted mean.

    Parameters
    ----------
    obj: xr.Dataset
        Input data
    time: str, optional
        Name of time coordinate

    Returns
    -------
    reduced object
    """
    time = _get_time(obj, time)

    season_obj = seasonal_weighted_mean(obj, time)
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        obj = obj.convert_calendar("noleap", align_on="date")
        month_length = obj[time].dt.days_in_month
        weights = month_length.groupby(f"{time}.season").sum() / (
            month_length.groupby(f"{time}.season").sum().sum()
        )
        obj = (season_obj * weights).sum(dim="season") / weights.sum("season")
    return obj


def _spatial_weighted_reduction(
    obj: xr.Dataset | xr.DataArray,
    func: str,
    lon: str | None = None,
    lat: str | None = None,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    lon, lat = _get_lon_and_lat(obj, lon, lat)
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = _spatial_weights(obj, lon, lat)
        obj = getattr(obj.weighted(weights), func)(dim=(lon, lat), **kwargs)
    return obj


def spatial_weighted_mean(
    obj: xr.Dataset | xr.DataArray, lon: str | None = None, lat: str | None = None
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    return _spatial_weighted_reduction(obj, "mean", lon, lat)


def spatial_weighted_std(
    obj: xr.Dataset | xr.DataArray, lon: str | None = None, lat: str | None = None
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial std with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    return _spatial_weighted_reduction(obj, "std", lon, lat)


def spatial_weighted_median(
    obj: xr.Dataset | xr.DataArray, lon: str | None = None, lat: str | None = None
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial median with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    return _spatial_weighted_reduction(obj, "quantile", lon, lat, q=0.5)


def spatial_weighted_statistics(
    obj: xr.Dataset | xr.DataArray, lon: str | None = None, lat: str | None = None
) -> xr.Dataset | xr.DataArray:
    objects = []
    for func in (spatial_weighted_mean, spatial_weighted_std, spatial_weighted_median):
        objects.append(
            func(obj, lon, lat).expand_dims(
                statistic=[func.__name__.replace("spatial_weighted_", "")]
            )
        )
    ds = xr.merge(objects)
    if isinstance(obj, xr.DataArray):
        return ds[obj.name]
    return ds


def _spatial_weighted_rmse(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon: str | None = None,
    lat: str | None = None,
    centralise: bool = False,
) -> xr.Dataset | xr.DataArray:
    lon, lat = _get_lon_and_lat(obj1, lon, lat)
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = _spatial_weights(obj1, lon, lat)
        if centralise:
            obj1 -= spatial_weighted_mean(obj1)
            obj2 -= spatial_weighted_mean(obj2)
        obj = (obj1 - obj2) ** 2
        return obj.weighted(weights).mean((lon, lat)) ** 0.5


def spatial_weighted_rmse(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon: str | None = None,
    lat: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial rmse with latitude weighting.

    Parameters
    ----------
    obj1, obj2: xr.Dataset or xr.DataArray
        Input data
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    return _spatial_weighted_rmse(obj1, obj2, lon, lat, centralise=False)


def spatial_weighted_crmse(
    obj1: xr.Dataset | xr.DataArray,
    obj2: xr.Dataset | xr.DataArray,
    lon: str | None = None,
    lat: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial crmse with latitude weighting.

    Parameters
    ----------
    obj1, obj2: xr.Dataset or xr.DataArray
        Input data
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    return _spatial_weighted_rmse(obj1, obj2, lon, lat, centralise=True)
