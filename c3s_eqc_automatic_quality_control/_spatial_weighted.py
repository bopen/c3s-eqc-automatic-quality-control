import dataclasses
import functools
from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr
from typing_extensions import TypedDict
from xarray.typing import DataArrayWeighted, DatasetWeighted

from . import utils


class SpatialWeightedKwargs(TypedDict):
    lon_name: Hashable
    lat_name: Hashable
    weights: xr.DataArray | bool


@dataclasses.dataclass
class SpatialWeighted:
    obj: xr.DataArray | xr.Dataset
    lon_name: Hashable | None
    lat_name: Hashable | None
    weights: xr.DataArray | bool

    @functools.cached_property
    def kwargs(self) -> SpatialWeightedKwargs:
        return SpatialWeightedKwargs(
            lon_name=self.lon.name,
            lat_name=self.lat.name,
            weights=False if self.weights is False else self.obj_weighted.weights,
        )

    @functools.cached_property
    def lon(self) -> xr.DataArray:
        return xr.DataArray(
            self.obj[
                self.lon_name
                if self.lon_name is not None
                else utils.get_coord_name(self.obj, "longitude")
            ]
        )

    @functools.cached_property
    def lat(self) -> xr.DataArray:
        return xr.DataArray(
            self.obj[
                self.lat_name
                if self.lat_name is not None
                else utils.get_coord_name(self.obj, "latitude")
            ]
        )

    @functools.cached_property
    def reduction_dims(self) -> list[str]:
        dims = set(self.lon.dims) | set(self.lat.dims)
        return sorted(map(str, dims))

    @functools.cached_property
    def obj_weighted(
        self,
    ) -> xr.DataArray | xr.Dataset | DataArrayWeighted | DatasetWeighted:
        if isinstance(self.weights, xr.DataArray):
            return self.obj.weighted(self.weights)
        if self.weights is True:
            return self.obj.weighted(np.abs(np.cos(np.deg2rad(self.lat))))
        return self.obj

    @utils.keep_attrs
    def reduce(self, func_name: str, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        if "dim" not in kwargs:
            kwargs["dim"] = self.reduction_dims
        obj: xr.DataArray | xr.Dataset = getattr(self.obj_weighted, func_name)(**kwargs)
        return obj

    @utils.keep_attrs
    def centralise(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        return self.obj - self.reduce("mean", **kwargs)

    @utils.keep_attrs
    def rmse(
        self, obj: xr.DataArray | xr.Dataset, centralise: bool = False, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        if centralise:
            weighted_obj = SpatialWeighted(obj, **self.kwargs)
            diff = weighted_obj.centralise(**kwargs) - self.centralise(**kwargs)
        else:
            diff = obj - self.obj
        weighted_diff2 = SpatialWeighted(diff**2, **self.kwargs)
        return weighted_diff2.reduce("mean", **kwargs) ** 0.5

    @utils.keep_attrs
    def corr(
        self, obj: xr.DataArray | xr.Dataset, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        weighted_obj = SpatialWeighted(obj, **self.kwargs)
        central_prod = self.centralise(**kwargs) * weighted_obj.centralise(**kwargs)
        num = SpatialWeighted(central_prod, **self.kwargs).reduce("mean", **kwargs)
        den = self.reduce("std", **kwargs) * weighted_obj.reduce("std", **kwargs)
        return num / den
