import dataclasses
import functools
from typing import Any, Callable, Hashable, TypeVar

import numpy as np
import xarray as xr
from typing_extensions import ParamSpec, TypedDict
from xarray.core.weighted import DataArrayWeighted, DatasetWeighted

P = ParamSpec("P")
T = TypeVar("T")


def keep_attrs(func: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with xr.set_options(keep_attrs=True):  # type: ignore[no-untyped-call]
            return func(*args, **kwargs)

    return wrapper


SpatialWeightedKwargs = TypedDict(
    "SpatialWeightedKwargs",
    {"lon_name": Hashable, "lat_name": Hashable, "weights": xr.DataArray},
)


@dataclasses.dataclass
class SpatialWeighted:
    obj: xr.DataArray | xr.Dataset
    lon_name: Hashable | None
    lat_name: Hashable | None
    weights: xr.DataArray | None

    def get_coord(self, coordinate: str) -> Hashable:
        coords: list[Hashable] = self.obj.cf.coordinates[coordinate]
        if isinstance(self.obj, xr.Dataset):
            bounds = self.obj.cf.bounds.get(coordinate, [])
            coords = list(set(coords) - set(bounds))
        if len(coords) == 1:
            return coords[0]
        raise ValueError(f"Can not infer {coordinate!r}: {coords!r}")

    @functools.cached_property
    def kwargs(self) -> SpatialWeightedKwargs:
        return SpatialWeightedKwargs(
            lon_name=self.lon.name,
            lat_name=self.lat.name,
            weights=self.obj_weighted.weights,
        )

    @functools.cached_property
    def lon(self) -> xr.DataArray:
        return xr.DataArray(self.obj[self.lon_name or self.get_coord("longitude")])

    @functools.cached_property
    def lat(self) -> xr.DataArray:
        return xr.DataArray(self.obj[self.lat_name or self.get_coord("latitude")])

    @functools.cached_property
    def reduction_dims(self) -> list[str]:
        dims = set(self.lon.dims) | set(self.lat.dims)
        return sorted(map(str, dims))

    @functools.cached_property
    def obj_weighted(self) -> DataArrayWeighted | DatasetWeighted:
        weights: xr.DataArray = (
            self.weights if self.weights is not None else np.cos(np.deg2rad(self.lat))
        )
        return self.obj.weighted(weights)

    @keep_attrs
    def reduce(self, func: str, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        if "dim" not in kwargs:
            kwargs["dim"] = self.reduction_dims
        obj: xr.DataArray | xr.Dataset = getattr(self.obj_weighted, func)(**kwargs)
        return obj

    @keep_attrs
    def centralise(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        return self.obj - self.reduce("mean", **kwargs)

    @keep_attrs
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

    @keep_attrs
    def corr(
        self, obj: xr.DataArray | xr.Dataset, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        weighted_obj = SpatialWeighted(obj, **self.kwargs)
        central_prod = self.centralise(**kwargs) * weighted_obj.centralise(**kwargs)
        num = SpatialWeighted(central_prod, **self.kwargs).reduce("mean", **kwargs)
        den = self.reduce("std", **kwargs) * weighted_obj.reduce("std", **kwargs)
        return num / den
