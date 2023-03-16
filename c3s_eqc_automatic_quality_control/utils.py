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
    if lon_slice.start is None and lon_slice.stop is None:
        pass
    elif lon_slice.start or lon_slice.stop < 0:
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            obj[lon_name] = (obj[lon_name] + 180) % 360 - 180
    else:
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            obj[lon_name] %= 360
    obj = obj.sortby([lon_name, lat_name])
    return obj.sel({lon_name: lon_slice, lat_name: lat_slice})
