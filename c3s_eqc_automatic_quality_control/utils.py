"""Utilities."""

# Copyright 2023, European Union.
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
import xarray as xr


def _get_time(obj: xr.Dataset | xr.DataArray, time: str | None) -> str:
    if time is None:
        (time,) = obj.cf.coordinates["time"]
    return time


def _get_lon_and_lat(
    obj: xr.Dataset | xr.DataArray, lon: str | None, lat: str | None
) -> tuple[str, str]:
    if lon is None:
        (lon,) = obj.cf.coordinates["longitude"]
    if lat is None:
        (lat,) = obj.cf.coordinates["latitude"]
    return lon, lat


def regionalise(
    obj: xr.Dataset | xr.DataArray,
    lon_slice: slice = slice(None, None, None),
    lat_slice: slice = slice(None, None, None),
    lon_name: str | None = None,
    lat_name: str | None = None,
) -> xr.Dataset | xr.DataArray:
    """Extract a region cutout.

    Parameters
    ----------
    obj: xr.Dataset | xr.DataArray
        Object to cut
    lon_slice, lat_slice: slice
        Coordinate slices defining the region to cutout
    lon_name, lat_name: str
        Name of longitude/latitude coordinate

    Returns
    -------
    Cutout object
    """
    lon_name, lat_name = _get_lon_and_lat(obj, lon_name, lat_name)
    indexers = {lon_name: lon_slice, lat_name: lat_slice}

    # Convert longitude
    lon_limits = xr.DataArray([lon_slice.start, lon_slice.stop], dims=lon_name)
    lon_limits = lon_limits.dropna(lon_name)
    if (lon_limits < 0).any() and (obj[lon_name] >= 0).all():
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            obj[lon_name] = (obj[lon_name] + 180) % 360 - 180
        obj[lon_name] = obj[lon_name].sortby(lon_name)
    elif (lon_limits > 180).any() and (obj[lon_name] <= 180).all():
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            obj[lon_name] = obj[lon_name] % 360
        obj[lon_name] = obj[lon_name].sortby(lon_name)

    # Sort
    for name, slice in indexers.items():
        bounds = obj[name][[0, -1]]
        ascending_bounds = bool(bounds.diff(name) > 0)
        ascending_slice = bool(
            xr.DataArray([slice.start, slice.stop], dims=name).fillna(bounds).diff(name)
            > 0
        )
        if ascending_bounds is not ascending_slice:
            obj = obj.sortby(name, ascending=ascending_slice)
    return obj.sel(indexers)
