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


def spatial_mean(
    ds: xr.Dataset | xr.DataArray, lon: str = "longitude", lat: str = "latitude"
) -> xr.Dataset | xr.DataArray:
    """
    Calculate spatial mean of ds with latitude weighting.

    Parameters
    ----------
    ds: xr.Dataset or xr.DataArray
        Input data on which to apply the spatial mean
    lon: str, optional
        Name of longitude coordinate
    lat: str, optional
        Name of latitude coordinate
    Returns
    -------
    ds spatially mean
    """
    cos = np.cos(np.deg2rad(ds[lat]))
    weights = cos / (cos.sum(lat) * len(ds[lon]))
    return (ds * weights).sum(dim=[lon, lat])  # type: ignore


def spatial_daily_mean(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    ds = spatial_mean(ds)
    return ds.resample(time="1D").mean("time")
