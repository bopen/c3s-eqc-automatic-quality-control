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
import contextlib
import datetime
import functools
import itertools
import os
import pathlib
from collections.abc import Callable, Iterator
from typing import Any

import cacholote
import cf_xarray  # noqa: F401
import cgul
import earthkit.data
import fsspec
import fsspec.implementations.local
import joblib
import pandas as pd
import tqdm
import xarray as xr
from earthkit.data.readers.csv import CSVReader
from earthkit.data.readers.grib.index import GribFieldList
from earthkit.data.sources.file import File

N_JOBS = 1
INVALIDATE_CACHE = False
NOCACHE = False
_SORTED_REQUEST_PARAMETERS = ("area", "grid")


@contextlib.contextmanager
def _set_env(**kwargs: Any) -> Iterator[None]:
    old_environ = dict(os.environ)
    try:
        os.environ.update(
            {k.upper(): str(v) for k, v in kwargs.items() if v is not None}
        )
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


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
    if isinstance(obj, list | tuple | set | range):
        return list(obj)
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
        chunks = {
            k: 1
            for k, v in request.items()
            if isinstance(v, tuple | list | set) and k not in _SORTED_REQUEST_PARAMETERS
        }

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
        v = ensure_list(v)
        if k not in _SORTED_REQUEST_PARAMETERS:
            v = sorted(v)
        cacheable_request[k] = v[0] if len(v) == 1 else v
    return cacheable_request


def get_paths(sources: list[Any]) -> list[str]:
    paths = []
    for source in sources:
        indexes = getattr(source, "_indexes", [source])
        paths.extend([index.path for index in indexes])
    return paths


@cacholote.cacheable
def _cached_retrieve(
    collection_id: str, request: dict[str, Any]
) -> list[fsspec.implementations.local.LocalFileOpener]:
    if NOCACHE:
        request = request | {"nocache": datetime.datetime.now().isoformat()}
    ds = earthkit.data.from_source("cds", collection_id, request, prompt=False)
    sources = ds.sources if hasattr(ds, "sources") else [ds]
    fs = fsspec.filesystem("file")
    return [fs.open(path) for path in get_paths(sources)]


def retrieve(collection_id: str, request: dict[str, Any]) -> list[str]:
    with cacholote.config.set(
        return_cache_entry=False,
        io_delete_original=True,
    ):
        return [file.path for file in _cached_retrieve(collection_id, request)]


def get_sources(
    collection_id: str,
    request_list: list[dict[str, Any]],
) -> list[str]:
    sources: set[str] = set()
    disable = os.getenv("TQDM_DISABLE", "False") == "True"
    for request in tqdm.tqdm(request_list, disable=disable):
        sources.update(retrieve(collection_id, request))
    return list(sources)


def _set_bound_coords(ds: xr.Dataset) -> xr.Dataset:
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


def _set_time_dim(ds: xr.Dataset, collection_id: str) -> xr.Dataset:
    # TODO: Some dataset are missing the time dimension, so they can not be squeezed
    if collection_id == "satellite-earth-radiation-budget":
        if "time" in ds:
            try:
                ds["time"].dt
            except TypeError:
                # Drop as it's not a time variable
                ds = ds.squeeze("time", drop=True)

    cf_time_dims = set(ds.cf.coordinates.get("time", [])) & set(ds.dims)
    if not ("time" in ds.dims or cf_time_dims):
        for time in ("forecast_reference_time", "time"):
            if time in ds.variables and len(ds[time].dims) <= 1:
                if not ds[time].dims:
                    ds = ds.expand_dims(time)
                else:
                    # E.g., satellite-methane
                    ds = ds.swap_dims({ds[time].dims[0]: time})
                break
        else:
            if collection_id.startswith("satellite-") and "source" in ds.encoding:
                # E.g., satellite-aerosol-properties
                ds = ds.expand_dims(source=[pathlib.Path(ds.encoding["source"]).stem])
    return ds


def _set_pressure_coord(ds: xr.Dataset) -> xr.Dataset:
    # TODO: Some satellite data have dimension "pressure" but coordinate "pre"
    # E.g., satellite-carbon-dioxide
    if "pre" in ds.variables:
        da = ds["pre"]
        if set(da.dims) == {"pressure"} and "pressure" not in ds.variables:
            ds = ds.assign_coords(pressure=da)
        ds = ds.drop_vars("pre")
    return ds


def harmonise(ds: xr.Dataset, collection_id: str) -> xr.Dataset:
    ds = cgul.harmonise(ds)
    # TODO: Various workarounds that cgul should eventually handle
    ds = _set_pressure_coord(ds)
    ds = _set_time_dim(ds, collection_id)
    ds = _set_bound_coords(ds)
    return ds


def _preprocess(
    ds: xr.Dataset,
    collection_id: str,
    preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset:
    if preprocess is not None:
        ds = preprocess(ds)
    return harmonise(ds, collection_id)


def _download_and_transform_requests(
    collection_id: str,
    request_list: list[dict[str, Any]],
    transform_func: Callable[..., xr.Dataset] | None,
    transform_func_kwargs: dict[str, Any],
    **open_mfdataset_kwargs: Any,
) -> xr.Dataset:
    sources = get_sources(collection_id, request_list)
    preprocess = functools.partial(
        _preprocess,
        collection_id=collection_id,
        preprocess=open_mfdataset_kwargs.pop("preprocess", None),
    )

    if all(
        isinstance(source, str) and source.endswith((".grib", ".grb", ".grb1", ".grb2"))
        for source in sources
    ):
        open_mfdataset_kwargs["preprocess"] = preprocess
        ds = xr.open_mfdataset(sources, **open_mfdataset_kwargs)
    else:
        ek_ds = earthkit.data.from_source("file", sources)
        if isinstance(ek_ds, GribFieldList):
            # https://github.com/ecmwf/earthkit-data/issues/374
            # squeeze=True is cfgrib default
            open_dataset_kwargs = {
                "chunks": {},
                "squeeze": True,
            } | open_mfdataset_kwargs
            ds = ek_ds.to_xarray(xarray_open_dataset_kwargs=open_dataset_kwargs)
            ds = preprocess(ds)
        elif isinstance(ek_ds, File) and isinstance(ek_ds._reader, CSVReader):
            assert not open_mfdataset_kwargs
            ds = preprocess(ek_ds.to_xarray())
        else:
            open_mfdataset_kwargs["preprocess"] = preprocess
            ds = ek_ds.to_xarray(xarray_open_mfdataset_kwargs=open_mfdataset_kwargs)
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"`earthkit.data` returned {type(ds)} instead of a xr.Dataset"
            )

    if transform_func is not None:
        with cacholote.config.set(return_cache_entry=False):
            ds = transform_func(ds, **transform_func_kwargs)
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"`transform_func` must return a xr.Dataset, while it returned {type(ds)}"
            )

    # TODO: make cacholote add coordinates? Needed to guarantee roundtrip
    # See: https://docs.xarray.dev/en/stable/user-guide/io.html#coordinates
    ds.attrs["coordinates"] = " ".join([str(coord) for coord in ds.coords])
    return ds


@joblib.delayed  # type: ignore[misc]
def _delayed_download(
    collection_id: str, request: dict[str, Any], config: cacholote.config.Settings
) -> None:
    with cacholote.config.set(**dict(config)):
        retrieve(collection_id, request)


def download_and_transform(
    collection_id: str,
    requests: list[dict[str, Any]] | dict[str, Any],
    chunks: dict[str, int] = {},
    split_all: bool = False,
    transform_func: Callable[..., xr.Dataset] | None = None,
    transform_func_kwargs: dict[str, Any] = {},
    transform_chunks: bool = True,
    n_jobs: int | None = None,
    invalidate_cache: bool | None = None,
    cached_open_mfdataset_kwargs: bool | dict[str, Any] = {},
    quiet: bool = False,
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
    n_jobs: int, optional
        Number of jobs for parallel download (download everything first)
        If None, use global variable N_JOBS
    invalidate_cache: bool, optional
        Whether to invalidate the cache entry or not.
        If None, use global variable INVALIDATE_CACHE
    cached_open_mfdataset_kwargs: bool | dict
        Kwargs to be passed on to xr.open_mfdataset for cached files.
        If True, use open_mfdataset_kwargs used for raw files.
    quiet: bool
        Whether to disable progress bars.
    **open_mfdataset_kwargs:
        Kwargs to be passed on to xr.open_mfdataset for raw files.

    Returns
    -------
    xr.Dataset
    """
    assert isinstance(quiet, bool)

    if n_jobs is None:
        n_jobs = N_JOBS

    if invalidate_cache is None:
        invalidate_cache = INVALIDATE_CACHE

    cached_open_mfdataset_kwargs = (
        open_mfdataset_kwargs
        if cached_open_mfdataset_kwargs is True
        else cached_open_mfdataset_kwargs or {}
    )

    use_cache = transform_func is not None
    func = functools.partial(
        cacholote.cacheable(_download_and_transform_requests)
        if use_cache
        else _download_and_transform_requests,
        collection_id=collection_id,
        transform_func=transform_func,
        transform_func_kwargs=transform_func_kwargs,
        **open_mfdataset_kwargs,
    )

    request_list = []
    for request in ensure_list(requests):
        request_list.extend(split_request(request, chunks, split_all))

    if invalidate_cache and not use_cache:
        # Delete raw data
        for request in request_list:
            cacholote.delete(
                _cached_retrieve, collection_id=collection_id, request=request
            )

    if n_jobs != 1:
        assert not NOCACHE, "n_jobs must be 1 when NOCACHE is True"
        # Download all data in parallel
        joblib.Parallel(n_jobs=n_jobs)(
            _delayed_download(collection_id, request, cacholote.config.get())
            for request in request_list
        )

    if use_cache and transform_chunks:
        # Cache each chunk transformed
        sources = []
        for request in tqdm.tqdm(request_list, disable=quiet):
            if invalidate_cache:
                cacholote.delete(
                    func.func, *func.args, request_list=[request], **func.keywords
                )
            with (
                cacholote.config.set(return_cache_entry=True),
                _set_env(tqdm_disable=True),
            ):
                sources.append(func(request_list=[request]).result["args"][0]["href"])
        ds = xr.open_mfdataset(sources, **cached_open_mfdataset_kwargs)
    else:
        # Cache final dataset transformed
        if invalidate_cache:
            cacholote.delete(
                func.func, *func.args, request_list=request_list, **func.keywords
            )
        with _set_env(tqdm_disable=quiet):
            ds = func(request_list=request_list)

    ds.attrs.pop("coordinates", None)  # Previously added to guarantee roundtrip
    return ds
