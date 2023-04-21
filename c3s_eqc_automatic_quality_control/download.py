"""C3S EQC Automatic Quality Control.

This module manages the execution of the quality control.
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

import calendar
import fnmatch
import functools
import itertools
import pathlib
from collections.abc import Callable
from typing import Any

import cacholote
import cads_toolbox
import cf_xarray  # noqa: F401
import cgul
import emohawk.readers.directory
import pandas as pd
import tqdm
import xarray as xr

from . import dashboard

cads_toolbox.config.USE_CACHE = True

LOGGER = dashboard.get_logger()


# In the future, this kwargs should somehow be handle upstream by the toolbox.
TO_XARRAY_KWARGS: dict[str, Any] = {
    "pandas_read_csv_kwargs": {"comment": "#"},
}


def compute_stop_date(switch_month_day: int | None = None) -> pd.Period:
    today = pd.Timestamp.today()
    if switch_month_day is None:
        switch_month_day = 9

    if today.day >= switch_month_day:
        date = today - pd.DateOffset(months=1)
    else:
        date = today - pd.DateOffset(months=2)
    return pd.Period(f"{date.year}-{date.month}")


def ceil_to_month(period: pd.Period, month: int = 1) -> pd.Period:
    if period.month > month:
        return pd.Period(year=period.year + 1, month=month, freq="M")
    if period.month < month:
        return pd.Period(year=period.year, month=month, freq="M")
    return period


def floor_to_month(period: pd.Period, month: int = 1) -> pd.Period:
    if period.month > month:
        return pd.Period(year=period.year, month=month, freq="M")
    if period.month < month:
        return pd.Period(year=period.year - 1, month=month, freq="M")
    return period


def extract_leading_months(start: pd.Period, stop: pd.Period) -> list[dict[str, Any]]:
    time_ranges = []
    if start.month > 1 and (start.year < stop.year or stop.month == 12):
        stop = min(stop, pd.Period(year=start.year, month=12, freq="M"))
        months = list(range(start.month, stop.month + 1))
        if len(months) > 0:
            time_ranges = [
                {
                    "year": start.year,
                    "month": months,
                    "day": list(range(1, 31 + 1)),
                }
            ]

    return time_ranges


def extract_trailing_months(start: pd.Period, stop: pd.Period) -> list[dict[str, Any]]:
    time_ranges = []
    if not stop.month == 12:
        start = max(start, floor_to_month(stop, month=1))
        months = list(range(start.month, stop.month + 1))
        if len(months) > 0:
            time_ranges = [
                {
                    "year": start.year,
                    "month": months,
                    "day": list(range(1, 31 + 1)),
                }
            ]

    return time_ranges


def extract_years(start: pd.Period, stop: pd.Period) -> list[dict[str, Any]]:
    start = ceil_to_month(start, month=1)
    stop = floor_to_month(stop, month=12)
    years = list(range(start.year, stop.year + 1))
    time_ranges = []
    if len(years) > 0:
        time_ranges = [
            {
                "year": years,
                "month": list(range(1, 12 + 1)),
                "day": list(range(1, 31 + 1)),
            }
        ]
    return time_ranges


def compute_request_date(
    start: pd.Period,
    stop: pd.Period | None = None,
    switch_month_day: int | None = None,
) -> list[dict[str, list[int] | int]]:
    if not stop:
        stop = compute_stop_date(switch_month_day)

    time_range = (
        extract_leading_months(start, stop)
        + extract_years(start, stop)
        + extract_trailing_months(start, stop)
    )
    return time_range


def update_request_date(
    request: dict[str, Any],
    start: str | pd.Period,
    stop: str | pd.Period | None = None,
    switch_month_day: int | None = None,
    stringify_dates: bool = False,
) -> list[dict[str, Any]]:
    """
    Return the requests defined by 'request' for the period defined by start and stop.

    Parameters
    ----------
    request: dict
        Parameters of the request
    start: str or pd.Period
        String {start_year}-{start_month} pd.Period with freq='M'
    stop: str or pd.Period
        Optional string {stop_year}-{stop_month} pd.Period with freq='M'
        If None the stop date is computed using the `switch_month_day`
    switch_month_day: int
        Used to compute the stop date in case stop is None. The stop date is computed as follows:
        if current day > switch_month_day then stop_month = current_month - 1
        else stop_month = current_month - 2
    stringify_dates: bool
        Whether to convert date to strings

    Returns
    -------
    xr.Dataset: request or list of requests updated
    """
    start = pd.Period(start, "M")
    if stop is None:
        stop = compute_stop_date(switch_month_day=switch_month_day)
    else:
        stop = pd.Period(stop, "M")

    dates = compute_request_date(start, stop, switch_month_day=switch_month_day)
    requests = []

    for d in dates:
        padded_d = {}
        if stringify_dates:
            for key, value in d.items():
                if key in ("year", "month", "day"):
                    padded_d[key] = (
                        f"{value:02d}"
                        if isinstance(value, int)
                        else [f"{v:02d}" for v in value]
                    )
        requests.append({**request, **d, **padded_d})
    return requests


def ensure_list(obj: Any) -> list[Any]:
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def check_non_empty_date(request: dict[str, Any]) -> bool:
    ymd = ("year", "month", "day")
    if not set(ymd) <= set(request):
        # Not a date request
        return True

    date = itertools.product(*(ensure_list(request[key]) for key in ymd))
    for year, month, day in date:
        n_days = calendar.monthrange(int(year), int(month))[1]
        if int(day) <= n_days:
            return True
    return False


def build_chunks(
    values: list[Any] | Any,
    chunks_size: int,
) -> list[list[Any]] | list[Any]:
    values = ensure_list(values)
    values.copy()
    if chunks_size == 1:
        return values
    else:
        chunks_list: list[list[Any]] = []
        for k, value in enumerate(values):
            if k % chunks_size == 0:
                chunks_list.append([])
            chunks_list[-1].append(value)
        return chunks_list


def split_request(
    request: dict[str, Any], chunks: dict[str, int] = {}, split_all: bool = False
) -> list[dict[str, Any]]:
    """
    Split the input request in smaller request defined by the chunks.

    Parameters
    ----------
    request: dict
        Parameters of the request
    chunks: dict
        Dictionary: {parameter_name: chunk_size}
    split_all: bool
        Split all parameters. Mutually exclusive with chunks

    Returns
    -------
    xr.Dataset
    """
    if chunks and split_all:
        raise ValueError("`chunks` and `split_all` are mutually exclusive")
    if split_all:
        chunks = {k: 1 for k, v in request.items() if isinstance(v, (tuple, list, set))}

    requests = []
    if not chunks:
        requests.append(request)
    else:
        list_values = list(
            itertools.product(
                *[
                    build_chunks(request[par], chunk_size)
                    for par, chunk_size in chunks.items()
                ]
            )
        )
        for values in list_values:
            out_request = request.copy()
            for parameter, value in zip(chunks, values):
                out_request[parameter] = value

            if not check_non_empty_date(out_request):
                continue

            requests.append(out_request)
    return [ensure_request_gets_cached(request) for request in requests]


def ensure_request_gets_cached(request: dict[str, Any]) -> dict[str, Any]:
    cacheable_request = {}
    for k, v in sorted(request.items()):
        if not isinstance(v, str):
            try:
                v = v[0] if len(v) == 1 else list(v)
            except TypeError:
                pass
        cacheable_request[k] = v
    return cacheable_request


def get_sources(
    collection_id: str,
    request_list: list[dict[str, Any]],
    exclude: list[str] = ["*.png", "*.json"],
) -> list[str]:
    source: set[str] = set()

    for request in request_list if len(request_list) == 1 else tqdm.tqdm(request_list):
        data = cads_toolbox.catalogue.retrieve(collection_id, request).data
        if content := getattr(data, "_content", None):
            source.update(map(str, content))
        else:
            source.add(str(data.source))

    for pattern in exclude:
        source -= set(fnmatch.filter(source, pattern))
    return list(source)


def _add_bounds(ds: xr.Dataset) -> xr.Dataset:
    # TODO: cgul should make bounds coordinates
    bounds = set(sum(ds.cf.bounds.values(), []))
    bounds.update(
        {
            var
            for var, da in ds.data_vars.items()
            if set(da.dims) & {"bnds", "bounds", "vertices"}
        }
    )
    ds = ds.set_coords(bounds)
    for var_names in ds.cf.coordinates.values():
        if len(var_names) == 2:
            coord_name, bound_name = sorted(var_names, key=lambda x: len(ds[x].dims))
            if len(set(ds[bound_name].dims) - set(ds[coord_name].dims)) == 1:
                ds[coord_name].attrs.setdefault("bounds", bound_name)
    return ds


def harmonise(ds: xr.Dataset) -> xr.Dataset:
    # TODO: additional rules to cgul
    ds = cgul.harmonise(ds)

    # Some satellite data have dimension "pressure" but coordinate "pre"
    if "pre" in ds.data_vars:
        da = ds["pre"]
        if set(da.dims) == {"pressure"} and "pressure" not in ds.variables:
            ds = ds.assign_coords(pressure=da)
        ds = ds.drop("pre")

    return ds


def _preprocess(
    ds: xr.Dataset,
    collection_id: str,
    preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset:
    if preprocess is not None:
        ds = preprocess(ds)

    ds = harmonise(ds)

    # TODO: workaround: sometimes single timestamps are squeezed
    cf_time_dims = set(ds.cf.coordinates.get("time", [])) & set(ds.dims)
    if not ("time" in ds.dims or cf_time_dims):
        if "forecast_reference_time" in ds.cf:
            ds = ds.cf.expand_dims("forecast_reference_time")
        elif "time" in ds:
            ds = ds.expand_dims("time")
        elif collection_id.startswith("satellite-") and "source" in ds.encoding:
            ds = ds.expand_dims(source=[pathlib.Path(ds.encoding["source"]).stem])

    ds = _add_bounds(ds)
    return ds


def get_data(source: list[str]) -> Any:
    if len(source) == 1:
        return emohawk.open(source[0])

    # TODO: emohawk not able to open a list of files
    emohwak_dir = emohawk.readers.directory.DirectoryReader("")
    emohwak_dir._content = source
    return emohwak_dir


def _download_and_transform_requests(
    collection_id: str,
    request_list: list[dict[str, Any]],
    transform_func: Callable[..., xr.Dataset] | None,
    transform_func_kwargs: dict[str, Any],
    **open_mfdataset_kwargs: Any,
) -> xr.Dataset:
    with cacholote.config.set(return_cache_entry=False):
        # TODO: Ideally, we would always use emohawk.
        # However, there is not a consistent behavior across backends.
        # For example, GRIB silently ignore open_mfdataset_kwargs
        sources = get_sources(collection_id, request_list)
        try:
            engine = open_mfdataset_kwargs.get(
                "engine",
                {xr.backends.plugins.guess_engine(source) for source in sources},
            )
            use_emohawk = len(engine) != 1
        except ValueError:
            use_emohawk = True

        open_mfdataset_kwargs["preprocess"] = functools.partial(
            _preprocess,
            collection_id=collection_id,
            preprocess=open_mfdataset_kwargs.get("preprocess", None),
        )

        if use_emohawk:
            data = get_data(sources)
            ds: xr.Dataset = data.to_xarray(
                xarray_open_mfdataset_kwargs=open_mfdataset_kwargs,
                **TO_XARRAY_KWARGS,
            )
            if not isinstance(ds, xr.Dataset):
                # When emohawk fails to concat, it silently return a list
                raise TypeError(
                    f"`emohawk` returned {type(ds)} instead of a xr.Dataset"
                )
        else:
            ds = xr.open_mfdataset(sources, **open_mfdataset_kwargs)

        if transform_func is not None:
            ds = transform_func(ds, **transform_func_kwargs)
            if not isinstance(ds, xr.Dataset):
                raise TypeError(
                    f"`transform_func` must return a xr.Dataset, while it returned {type(ds)}"
                )

        # TODO: make cacholote add coordinates? Needed to guarantee roundtrip
        # See: https://docs.xarray.dev/en/stable/user-guide/io.html#coordinates
        ds.attrs["coordinates"] = " ".join([str(coord) for coord in ds.coords])
        return ds


def download_and_transform(
    collection_id: str,
    requests: list[dict[str, Any]] | dict[str, Any],
    chunks: dict[str, int] = {},
    split_all: bool = False,
    transform_func: Callable[..., xr.Dataset] | None = None,
    transform_func_kwargs: dict[str, Any] = {},
    transform_chunks: bool = True,
    **open_mfdataset_kwargs: Any,
) -> xr.Dataset:
    """
    Download and transform data caching the results.

    Datasets are chunked along the parameters specified by `chunks`.
    If `transform_chunks` is True, the transform function is applied to each chunk.
    Otherwise, the transform function is applied to the whole dataset after merging all chunks.

    Parameters
    ----------
    collection_id: str
        ID of the dataset.
    requests: list of dict or dict
        Parameters of the requests
    chunks: dict
        Dictionary: {parameter_name: chunk_size}
    split_all: bool
        Split all parameters. Mutually exclusive with chunks
    transform_func: callable, optional
        Function to apply to each single chunk
    transform_func_kwargs: dict
        Kwargs to be passed on to `transform_func`
    transform_chunks: bool
        Whether to transform and cache each chunk or the whole dataset
    **open_mfdataset_kwargs:
        Kwargs to be passed on to xr.open_mfdataset

    Returns
    -------
    xr.Dataset
    """
    download_and_transform_requests = _download_and_transform_requests
    if is_cached := transform_func is not None:
        download_and_transform_requests = cacholote.cacheable(
            download_and_transform_requests
        )

    # Split requests
    request_list = []
    for request in ensure_list(requests):
        request_list.extend(split_request(request, chunks, split_all))

    if not transform_chunks or not is_cached:
        ds = download_and_transform_requests(
            collection_id,
            request_list,
            transform_func,
            transform_func_kwargs,
            **open_mfdataset_kwargs,
        )
    else:
        # Cache each chunk separately
        sources = []
        for request in tqdm.tqdm(request_list):
            with cacholote.config.set(
                return_cache_entry=True, raise_all_encoding_errors=True
            ):
                cache_entry = download_and_transform_requests(
                    collection_id,
                    [request],
                    transform_func,
                    transform_func_kwargs,
                    **open_mfdataset_kwargs,
                )
            sources.append(cache_entry.result["args"][0]["href"])
        open_mfdataset_kwargs.pop("preprocess", None)  # Already preprocessed
        ds = xr.open_mfdataset(sources, **open_mfdataset_kwargs)

    ds.attrs.pop("coordinates", None)
    return ds
