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

from typing import Union

import numpy as np
import xarray as xr

LONS = ("longitude", "lon")
LATS = ("latitude", "lat")


def get_coord_name(obj, valid_values):
    coords = [var for var in obj.coords if var.lower() in valid_values]
    if len(coords) == 1:
        return coords[0]
    raise ValueError("can NOT infer coordinate names")


def get_lon_lat_names(
    obj: Union[xr.Dataset, xr.DataArray], lon: str | None = None, lat: str | None = None
) -> tuple[str, str]:
    return (lon or get_coord_name(obj, LONS), lat or get_coord_name(obj, LATS))


def spatial_weights(
    obj: Union[xr.Dataset, xr.DataArray], lon: str | None = None, lat: str | None = None
):
    lon, lat = get_lon_lat_names(obj, lon, lat)
    cos = np.cos(np.deg2rad(obj[lat]))
    return cos / (cos.sum(lat) * len(obj[lon]))


def spatial_weighted_mean(
    obj: Union[xr.Dataset, xr.DataArray], lon: str | None = None, lat: str | None = None
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data on which to apply the spatial mean
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    lon, lat = get_lon_lat_names(obj, lon, lat)
    with xr.set_options(keep_attrs=True):
        weights = spatial_weights(obj, lon, lat)
        return obj.weighted(weights).mean((lon, lat))


def spatial_weighted_std(
    obj: Union[xr.Dataset, xr.DataArray], lon: str | None = None, lat: str | None = None
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Calculate spatial std of ds with latitude weighting.

    Parameters
    ----------
    obj: xr.Dataset or xr.DataArray
        Input data on which to apply the spatial mean
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate

    Returns
    -------
    reduced object
    """
    lon, lat = get_lon_lat_names(obj, lon, lat)
    with xr.set_options(keep_attrs=True):
        weights = spatial_weights(obj, lon, lat)
        return obj.weighted(weights).std((lon, lat))
