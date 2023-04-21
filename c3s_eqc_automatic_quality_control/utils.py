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
from typing import Callable, Hashable, TypeVar

import xarray as xr
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def keep_attrs(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            return func(*args, **kwargs)

    return wrapper


def get_coord_name(obj: xr.DataArray | xr.Dataset, coordinate: str) -> Hashable:
    coords: list[Hashable] = obj.cf.coordinates.get(coordinate, [])
    if isinstance(obj, xr.Dataset):
        bounds = obj.cf.bounds.get(coordinate, [])
        coords = list(set(coords) - set(bounds))
    if coordinate == "time":
        coords = list(set(coords) & set(obj.dims))
    if len(coords) == 1:
        return coords[0]
    raise ValueError(f"Can NOT infer {coordinate}: {coords}")


@keep_attrs
def regionalise(
    obj: xr.Dataset | xr.DataArray,
    lon_slice: slice = slice(None, None, None),
    lat_slice: slice = slice(None, None, None),
    lon_name: Hashable | None = None,
    lat_name: Hashable | None = None,
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
    lon_name = lon_name if lon_name is not None else get_coord_name(obj, "longitude")
    lon_name = lat_name if lat_name is not None else get_coord_name(obj, "latitude")
    indexers = {lon_name: lon_slice, lat_name: lat_slice}

    # Convert longitude
    lon_limits = xr.DataArray([lon_slice.start, lon_slice.stop], dims=lon_name)
    lon_limits = lon_limits.dropna(lon_name)
    if (lon_limits < 0).any() and (obj[lon_name] >= 0).all():
        obj[lon_name] = (obj[lon_name] + 180) % 360 - 180
        obj[lon_name] = obj[lon_name].sortby(lon_name)
    elif (lon_limits > 180).any() and (obj[lon_name] <= 180).all():
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
