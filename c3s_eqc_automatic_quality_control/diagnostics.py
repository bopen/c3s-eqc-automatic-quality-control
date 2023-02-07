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


def _spatial_weights(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.DataArray:
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


def _regridder(
    grid_in: xr.Dataset, grid_out: xr.Dataset, method: str, **kwargs: Any
) -> xe.Regridder:
    # Remove metadate and cache using dicts
    dict_in = grid_in.cf[["longitude", "latitude"]].to_dict()
    dict_in.pop("attrs")
    dict_out = grid_out.cf[["longitude", "latitude"]].to_dict()
    dict_out.pop("attrs")

    kwargs["weights"] = _regridder_weights(dict_in, dict_out, method, **kwargs)
    return xe.Regridder(grid_in, grid_out, method, **kwargs)


def regrid(
    obj: xr.Dataset, grid_out: xr.Dataset, method: str, **kwargs: Any
) -> xr.Dataset:
    regridder = _regridder(obj, grid_out, method, **kwargs)
    obj = regridder(obj, keep_attrs=True)
    return obj


def spatial_weighted_mean(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data on which to apply the spatial mean
    lon: str
        Name of longitude coordinate
    lat: str
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = _spatial_weights(obj, lon, lat)
        return obj.weighted(weights).mean((lon, lat))


def seasonal_weighted_mean(obj: xr.Dataset, time: str = "time") -> xr.Dataset:
    """
    Calculate seasonal weighted mean.

    Parameters
    ----------
    obj: xr.Dataset
        Input data on which to apply the spatial mean
    time: str
        Name of time coordinate

    Returns
    -------
    reduced object
    """
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        obj = obj.convert_calendar("noleap", align_on="date")
        month_length = obj[time].dt.days_in_month
        weights = (
            month_length.groupby(f"{time}.season")
            / month_length.groupby(f"{time}.season").sum()
        )
        obj = (obj * weights).groupby(f"{time}.season").sum(dim=time)
        return obj


def spatial_weighted_std(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial std of ds with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data on which to apply the spatial mean
    lon: str
        Name of longitude coordinate
    lat: str
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = _spatial_weights(obj, lon, lat)
        return obj.weighted(weights).std((lon, lat))
