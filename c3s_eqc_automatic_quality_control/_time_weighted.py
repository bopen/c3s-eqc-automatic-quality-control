import dataclasses
import functools
from typing import Any, Hashable, TypeVar, overload

import xarray as xr
from typing_extensions import ParamSpec
from xarray.core.weighted import DataArrayWeighted, DatasetWeighted

from . import utils

P = ParamSpec("P")
T = TypeVar("T")


@dataclasses.dataclass
class TimeWeighted:
    obj: xr.DataArray | xr.Dataset
    time_name: Hashable | None
    weights: xr.DataArray | None

    def __post_init__(self) -> None:
        if self.weights is None:
            self.obj = self.obj.convert_calendar(
                "noleap",
                align_on="date",
                dim=str(
                    self.time_name
                    if self.time_name is not None
                    else utils.get_coord_name(self.obj, "time")
                ),
            )

    @functools.cached_property
    def time(self) -> xr.DataArray:
        return xr.DataArray(
            self.obj[
                self.time_name
                if self.time_name is not None
                else utils.get_coord_name(self.obj, "time")
            ]
        )

    @functools.cached_property
    @utils.keep_attrs
    def obj_weighted(self) -> DataArrayWeighted | DatasetWeighted:
        weights = (
            self.weights if self.weights is not None else self.time.dt.days_in_month
        )
        return self.obj.weighted(weights)

    @utils.keep_attrs
    def groupby_reduce(
        self, func_name: str, group: str, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        if "." not in group:
            group = ".".join([str(self.time.name), group])
        time_name, *_ = group.split(".", 1)
        obj = self.obj.assign_coords(__weights__=self.obj_weighted.weights)
        return obj.groupby(group).map(map_func, (time_name, func_name), **kwargs)

    @utils.keep_attrs
    def reduce(
        self, func_name: str, group: str | None = None, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        kwargs.setdefault("dim", self.obj_weighted.weights.dims)

        if group is not None:
            return self.groupby_reduce(func_name, group, **kwargs)
        obj: xr.DataArray | xr.Dataset = getattr(self.obj_weighted, func_name)(**kwargs)
        return obj


@overload
def map_func(
    grouped_obj: xr.DataArray,
    time_name: str,
    func: str,
    **kwargs: Any,
) -> xr.DataArray:
    ...


@overload
def map_func(
    grouped_obj: xr.Dataset,
    time_name: str,
    func: str,
    **kwargs: Any,
) -> xr.Dataset:
    ...


def map_func(
    grouped_obj: xr.DataArray | xr.Dataset,
    time_name: str,
    func: str,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    weights = xr.DataArray(grouped_obj["__weights__"])
    grouped_obj = grouped_obj.drop_vars("__weights__")
    return TimeWeighted(grouped_obj, time_name, weights).reduce(func, **kwargs)
