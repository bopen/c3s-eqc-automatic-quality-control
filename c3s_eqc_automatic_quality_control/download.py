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
import itertools
import logging
from collections.abc import Callable
from typing import Any

import cacholote
import cads_toolbox
import dask
import pandas as pd
import xarray as xr

from . import dashboard

LOGGER = dashboard.get_logger()


def compute_stop_date(switch_month_day: int | None = None) -> pd.Timestamp:
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
        period = pd.Period(year=period.year + 1, month=month, freq="M")
    if period.month < month:
        period = pd.Period(year=period.year, month=month, freq="M")
    return period


def floor_to_month(period: pd.Period, month: int = 1) -> pd.Period:

    if period.month > month:
        period = pd.Period(year=period.year, month=month, freq="M")
    if period.month < month:
        period = pd.Period(year=period.year - 1, month=month, freq="M")

    return period


def extract_leading_months(
    start: pd.Period, stop: pd.Period
) -> list[dict[str, list[int] | int]]:

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


def extract_trailing_months(
    start: pd.Period, stop: pd.Period
) -> list[dict[str, list[int] | int]]:

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


def extract_years(
    start: pd.Timestamp, stop: pd.Timestamp
) -> list[dict[str, list[int]]]:

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
    return time_range  # type: ignore


def update_request_date(
    request: dict[str, Any],
    start: str | pd.Period,
    stop: str | pd.Period | None = None,
    switch_month_day: int | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
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
    if isinstance(dates, dict):
        return {**request, **dates}
    requests = []
    for d in dates:
        requests.append({**request, **d})
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
    xr.Dataset: list of requests
    """
    if chunks and split_all:
        raise ValueError("`chunks` and `split_all` are mutually exclusive")
    if split_all:
        chunks = {k: 1 for k, v in request.items() if isinstance(v, (tuple, list, set))}

    if not chunks:
        return [request]

    requests = []
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
    return requests


@cacholote.cacheable
def download_and_transform_chunk(
    collection_id: str,
    request: dict[str, Any],
    func: None
    | (
        Callable[[xr.Dataset], xr.Dataset] | Callable[[pd.DataFrame], pd.DataFrame]
    ) = None,
    open_with: str = "xarray",
    logger: logging.Logger = LOGGER,
) -> xr.Dataset | pd.DataFrame:
    open_with_allowed_values = ("xarray", "pandas")
    if open_with not in open_with_allowed_values:
        raise ValueError(
            f"{open_with=} is not a valid value. Allowed values: {open_with_allowed_values!r}"
        )

    remote = cads_toolbox.catalogue.retrieve(collection_id, request)
    if open_with == "xarray":
        ds = remote.to_xarray(harmonise=True)
    elif open_with == "pandas":
        ds = remote.to_pandas()
    if func is not None:
        ds = func(ds)
    return ds


def download_and_transform(
    collection_id: str,
    requests: list[dict[str, Any]] | dict[str, Any],
    chunks: dict[str, int] = {},
    split_all: bool = False,
    func: None
    | (
        Callable[[xr.Dataset], xr.Dataset] | Callable[[pd.DataFrame], pd.DataFrame]
    ) = None,
    open_with: str = "xarray",
    logger: logging.Logger = LOGGER,
    **kwargs: Any,
) -> xr.Dataset | pd.DataFrame:
    """
    Download chunking along the selected parameters, apply the function f to each chunk and merge the results.

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
    func: callable
        Function to apply to each single chunk
    open_with: str
        Backend used for opening the data file, valid values: 'xarray', or 'pandas'
    **kwargs:
        kwargs to be passed on to xr.merge or pd.concat function

    Returns
    -------
    xr.Dataset or pd.DataFrame: Resulting dataset or dataframe.
    """
    request_list = []

    for request in ensure_list(requests):
        request_list.extend(split_request(request, chunks, split_all))
    datasets = []
    for n, request_chunk in enumerate(request_list):
        logger.info(f"Gathering file {n+1} out of {len(request_list)}...")
        ds = download_and_transform_chunk(
            collection_id, request=request_chunk, func=func, open_with=open_with
        )
        datasets.append(ds)
    logger.info("Aggregating data...")
    if open_with == "xarray":
        with dask.config.set({"array.slicing.split_large_chunks": True}):
            ds = xr.merge(datasets, **kwargs)
    else:
        ds = pd.concat(datasets, **kwargs)
    return ds
