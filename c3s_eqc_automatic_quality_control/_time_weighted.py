import dataclasses
import functools
from collections.abc import Hashable
from typing import Any, overload

import xarray as xr
from xarray.core.weighted import DataArrayWeighted, DatasetWeighted

from . import utils


@dataclasses.dataclass
class TimeWeighted:
    obj: xr.DataArray | xr.Dataset
    time_name: Hashable | None
    weights: xr.DataArray | bool

    def __post_init__(self) -> None:
        if self.weights is True:
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
    def obj_weighted(
        self,
    ) -> xr.DataArray | xr.Dataset | DataArrayWeighted | DatasetWeighted:
        if isinstance(self.weights, xr.DataArray):
            return self.obj.weighted(self.weights)
        if self.weights is True:
            return self.obj.weighted(self.time.dt.days_in_month)
        return self.obj

    @utils.keep_attrs
    def groupby_reduce(
        self, func_name: str, group: str, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        if "." not in group:
            group = ".".join([str(self.time.name), group])
        time_name, *_ = group.split(".", 1)
        obj = self.obj
        if isinstance(self.obj_weighted, (DataArrayWeighted | DatasetWeighted)):
            obj = obj.assign_coords(__weights__=self.obj_weighted.weights)
        return obj.groupby(group).map(map_func, (time_name, func_name), **kwargs)

    @utils.keep_attrs
    def reduce(
        self, func_name: str, group: str | None = None, **kwargs: Any
    ) -> xr.DataArray | xr.Dataset:
        if "dim" not in kwargs:
            kwargs["dim"] = self.time.name

        if group is not None:
            return self.groupby_reduce(func_name, group, **kwargs)
        obj: xr.DataArray | xr.Dataset = getattr(self.obj_weighted, func_name)(**kwargs)
        return obj

    def polyfit(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        if "w" not in kwargs:
            kwargs["w"] = (
                self.obj_weighted.weights
                if isinstance(self.obj_weighted, (DataArrayWeighted | DatasetWeighted))
                else None
            )
        if "dim" not in kwargs:
            kwargs["dim"] = self.time.name

        if "w" in kwargs:
            # TODO: https://github.com/pydata/xarray/issues/5644
            return self.obj.copy(deep=True).polyfit(**kwargs)
        return self.obj.polyfit(**kwargs)

    def coverage(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        if "dim" not in kwargs:
            kwargs["dim"] = self.time.name
        if isinstance(self.obj_weighted, (DataArrayWeighted | DatasetWeighted)):
            cov = (
                self.obj.notnull()
                * self.obj_weighted.weights
                / self.obj_weighted.weights.sum(**kwargs)
            )
        else:
            cov = self.obj.count(**kwargs) / self.obj.sizes[kwargs["dim"]]
        return cov * 100


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
    weights = grouped_obj.coords.get("__weights__", False)
    if weights is not False:
        grouped_obj = grouped_obj.drop_vars("__weights__")
    return TimeWeighted(grouped_obj, time_name, weights).reduce(func, **kwargs)
