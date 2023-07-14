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
from typing import Any, Callable

import pyproj
import xarray as xr
from xarray.core.common import DataWithCoords

from . import _grid_cell_area, _regrid, _spatial_weighted, _time_weighted, utils

__all__ = [
    "annual_weighted_mean",
    "annual_weighted_std",
    "grid_cell_area",
    "regrid",
    "rolling_weighted_filter",
    "seasonal_weighted_mean",
    "seasonal_weighted_std",
    "spatial_weighted_corr",
    "spatial_weighted_crmse",
    "spatial_weighted_errors",
    "spatial_weighted_mean",
    "spatial_weighted_median",
    "spatial_weighted_rmse",
    "spatial_weighted_statistics",
    "spatial_weighted_std",
    "time_weighted_coverage",
    "time_weighted_linear_trend",
    "time_weighted_mean",
    "time_weighted_std",
]


def _apply_attrs_func(
    obj_out: xr.Dataset | xr.DataArray,
    obj_in: xr.Dataset | xr.DataArray,
    attrs_func: Callable[[dict[str, Any]], dict[str, Any]],
) -> xr.Dataset | xr.DataArray:
    if isinstance(obj_out, xr.Dataset):
        for var, da in obj_out.data_vars.items():
            da.attrs = attrs_func(obj_in[var].attrs)
    else:
        obj_out.attrs = attrs_func(obj_in.attrs)
    return obj_out


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


def time_weighted_linear_trend(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    p_value: bool = False,
    **kwargs: Any,
) -> (
    xr.DataArray
    | xr.Dataset
    | tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]
):
    """
    Calculate time weighted linear trend.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    obj_tuple = _time_weighted.TimeWeighted(obj, time_name, weights).linear_trend(
        p_value=p_value, **kwargs
    )
    trend = obj_tuple[0] * 1.0e9  # 1/ns to 1/s

    def attrs_func(attrs: dict[str, Any]) -> dict[str, Any]:
        return {
            "long_name": f"Linear trend of {attrs.get('long_name', '')}",
            "units": f"{attrs.get('units', '')} s-1",
        }

    trend = _apply_attrs_func(trend, obj, attrs_func)
    return trend if not p_value else (trend, obj_tuple[1])


def time_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate time weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", None, **kwargs
    )


def time_weighted_std(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate time weighted std.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "std", None, **kwargs
    )


def time_weighted_coverage(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate time weighted coverage.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    coverage = _time_weighted.TimeWeighted(obj, time_name, weights).coverage(**kwargs)

    def attrs_func(attrs: dict[str, Any]) -> dict[str, Any]:
        return {
            "long_name": f"Normalized coverage of {attrs.get('long_name', '')}",
            "units": "1",
        }

    return _apply_attrs_func(coverage, obj, attrs_func)


def seasonal_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate seasonal weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", "season", **kwargs
    )


def seasonal_weighted_std(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate seasonal weighted std.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "std", "season", **kwargs
    )


def annual_weighted_mean(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate annual weighted mean.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "mean", "year", **kwargs
    )


def annual_weighted_std(
    obj: xr.DataArray | xr.Dataset,
    time_name: Hashable | None = None,
    weights: xr.DataArray | bool = True,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Calculate annual weighted std.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input data
    time_name: str, optional
        Name of time coordinate
    weights: DataArray, bool, default: True
        Weights to apply:
        - True: weights are the number of days in each month
        - False: unweighted
        - DataArray: custom weights

    Returns
    -------
    DataArray or Dataset
        Reduced object
    """
    return _time_weighted.TimeWeighted(obj, time_name, weights).reduce(
        "std", "year", **kwargs
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
    return _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights).median(
        **kwargs
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
        Object with mean, std, and median
    """
    sw = _spatial_weighted.SpatialWeighted(obj, lon_name, lat_name, weights)
    objects = []
    for func in ("mean", "std", "median"):
        if hasattr(sw, func):
            obj = getattr(sw, func)(**kwargs)
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
    obj1, obj2: Dataset or DataArray
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
    obj1, obj2: Dataset or DataArray
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
        Object with rmse, crmse, and correlation
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


@utils.keep_attrs
def rolling_weighted_filter(
    obj: xr.DataArray | xr.Dataset,
    weights_mapper: dict[str, Any] = {},
    **rolling_kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Apply a rolling weighted filter.

    Parameters
    ----------
    obj: DataArray or Dataset
        Input object
    weights_mapper: dict
        Dictionary mapping dimension to 1D weights. Weights are applied using `dot`.
        E.g., {"time": range(10)}.

    Returns
    -------
    DataArray or Dataset
        Reduced objects
    """
    rolling_kwargs.setdefault("center", True)

    weights_mapper = {
        k: xr.DataArray(v, dims=f"{k}_window") for k, v in weights_mapper.items()
    }
    ds = obj if isinstance(obj, xr.Dataset) else obj._to_temp_dataset()
    ds = (
        ds.rolling({k: v.size for k, v in weights_mapper.items()}, **rolling_kwargs)
        .construct({k: f"{k}_window" for k in weights_mapper})
        .map(xr.dot, args=weights_mapper.values())
    )
    if isinstance(obj, xr.DataArray):
        return obj._from_temp_dataset(ds)
    return ds
