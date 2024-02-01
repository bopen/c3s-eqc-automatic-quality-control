import dataclasses
import functools
from collections.abc import Hashable
from typing import Any, overload

import xarray as xr
import xskillscore as xs
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

        return self.obj.polyfit(**kwargs)

    def linear_trend(
        self, p_value: bool = False, r2: bool = False, **kwargs: Any
    ) -> dict[str, xr.DataArray | xr.Dataset]:
        coeff = self.polyfit(deg=1, **kwargs)
        if isinstance(self.obj, xr.DataArray):
            coeff = coeff["polyfit_coefficients"]
            coeff.name = self.obj.name or ""
        else:
            coeff = coeff.drop_vars(
                [
                    varname
                    for varname in map(str, coeff.data_vars)
                    if not varname.endswith("_polyfit_coefficients")
                ]
            )
            coeff = coeff.rename(
                {
                    varname: varname.replace("_polyfit_coefficients", "")
                    for varname in map(str, coeff.data_vars)
                }
            )
        obj_trend = coeff.sel(degree=1, drop=True)
        output = {"linear_trend": obj_trend}
        if not (p_value or r2):
            return output

        dim = kwargs.get("dim", self.time.name)
        weights = kwargs.get("w", self.obj_weighted.weights if self.weights else None)
        fit = xr.polyval(self.obj[dim], coeff)
        xs_kwargs = {
            "dim": dim,
            "weights": weights,
            **{k: v for k, v in kwargs.items() if k == "skipna"},
        }
        if p_value:
            output["p_value"] = xs.pearson_r_p_value(self.obj, fit, **xs_kwargs)
        if r2:
            output["r2"] = xs.r2(self.obj, fit, **xs_kwargs)
        return output

    def coverage(self, **kwargs: Any) -> xr.DataArray | xr.Dataset:
        if "dim" not in kwargs:
            kwargs["dim"] = self.time.name
        if isinstance(self.obj_weighted, (DataArrayWeighted | DatasetWeighted)):
            return (
                self.obj.notnull()
                * self.obj_weighted.weights
                / self.obj_weighted.weights.sum(**kwargs)
            )
        return self.obj.count(**kwargs) / self.obj.sizes[kwargs["dim"]]


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
