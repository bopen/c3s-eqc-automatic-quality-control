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

import numpy as np
import xarray as xr


def spatial_weights(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.DataArray:
    cos = np.cos(np.deg2rad(obj[lat]))
    weights: xr.DataArray = cos / (cos.sum(lat) * len(obj[lon]))
    return weights


def spatial_weighted_mean(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.Dataset | xr.DataArray:
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
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = spatial_weights(obj, lon, lat)
        return obj.weighted(weights).mean((lon, lat))


def spatial_weighted_std(
    obj: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.Dataset | xr.DataArray:
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
    with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
        weights = spatial_weights(obj, lon, lat)
        return obj.weighted(weights).std((lon, lat))
