import datetime
from typing import Any

import cads_toolbox
import pandas as pd
import pytest
import xarray as xr
from utils import mock_download

from c3s_eqc_automatic_quality_control import download

AIR_TEMPERATURE_REQUEST = (
    "era5-2mt-2019-03-uk.grib",
    {
        "time": ["2019-03-01T00", "2019-03-31T23"],
        "latitude": [58, 50],
        "longitude": [-10, 2],
    },
)


@pytest.mark.parametrize(
    "chunks,split_all,length",
    [
        ({"month": 1}, False, 12),
        ({"month": 1, "year": 1}, False, 4 * 12),
        ({}, True, 3 * 4 * 12),
    ],
)
def test_split_request(chunks: dict[str, int], split_all: bool, length: int) -> None:
    request = {
        "pressure_level": ["1", "2", "3"],
        "day": "01",
        "year": ["2019", "2020", "2021", "2022"],
        "month": list(range(1, 13)),
    }
    requests = download.split_request(request, chunks=chunks, split_all=split_all)
    assert len(requests) == length


def test_build_chunks() -> None:
    values = list(range(11))

    res = download.build_chunks(values, 1)
    assert res == values

    res = download.build_chunks(values, 11)
    assert res == [values]

    res = download.build_chunks(values, 3)
    assert res[-1] == [9, 10]


@pytest.mark.parametrize(
    "request_dict,non_empty",
    [
        ({"year": "2021", "month": ["01", "02", "03"], "day": ["30", "31"]}, True),
        ({"year": "2021", "month": "03", "day": "30"}, True),
        ({"year": "2021", "month": ["02", "04"], "day": "31"}, False),
        ({"year": "2021", "month": "02", "day": ["30", "31"]}, False),
    ],
)
def test_check_non_empty_date(request_dict: dict[str, Any], non_empty: bool) -> None:
    assert download.check_non_empty_date(request_dict) is non_empty


@pytest.mark.parametrize(
    "date,month,expected",
    [
        (pd.Period("2022-12", freq="M"), 1, pd.Period("2022-01", freq="M")),
        (pd.Period("2022-01", freq="M"), 1, pd.Period("2022-01", freq="M")),
        (pd.Period("2022-12", freq="M"), 12, pd.Period("2022-12", freq="M")),
        (pd.Period("2022-01", freq="M"), 12, pd.Period("2021-12", freq="M")),
    ],
)
def test_floor_to_month(date: pd.Period, month: int, expected: pd.Period) -> None:
    res = download.floor_to_month(date, month)
    assert res == expected


def test_extract_leading_months() -> None:
    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2023-06", freq="M")
    res = download.extract_leading_months(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == 2020
    assert res[0]["month"] == [6, 7, 8, 9, 10, 11, 12]

    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2023-12", freq="M")
    res = download.extract_leading_months(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == 2020
    assert res[0]["month"] == [6, 7, 8, 9, 10, 11, 12]

    # special case: if 'start' is the start of the year, then there are no leading months
    start = pd.Period("2020-01", freq="M")
    stop = pd.Period("2023-06", freq="M")
    res = download.extract_leading_months(start, stop)

    assert len(res) == 0

    # special cases: if start.year == stop.year, then the months are trailing months when possible.

    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2020-10", freq="M")
    res = download.extract_leading_months(start, stop)

    assert len(res) == 0

    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2020-12", freq="M")
    res = download.extract_leading_months(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == 2020
    assert res[0]["month"] == [6, 7, 8, 9, 10, 11, 12]


def test_extract_trailing_months() -> None:
    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2023-06", freq="M")
    res = download.extract_trailing_months(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == 2023
    assert res[0]["month"] == [1, 2, 3, 4, 5, 6]

    start = pd.Period("2020-01", freq="M")
    stop = pd.Period("2023-06", freq="M")
    res = download.extract_trailing_months(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == 2023
    assert res[0]["month"] == [1, 2, 3, 4, 5, 6]

    # special case: if 'stop' is the end of the year, then there are no trailing months
    start = pd.Period("2020-01", freq="M")
    stop = pd.Period("2023-12", freq="M")
    res = download.extract_trailing_months(start, stop)

    assert len(res) == 0

    # special cases: if start.year == stop.year, then the months are trailing months when possible.
    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2020-10", freq="M")
    res = download.extract_trailing_months(start, stop)

    assert len(res) == 1
    assert res[0]["month"] == [6, 7, 8, 9, 10]

    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2020-12", freq="M")
    res = download.extract_trailing_months(start, stop)

    assert len(res) == 0


def test_extract_years() -> None:
    start = pd.Period("2020-06", freq="M")
    stop = pd.Period("2023-06", freq="M")
    res = download.extract_years(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == [2021, 2022]

    start = pd.Period("2020-01", freq="M")
    stop = pd.Period("2020-12", freq="M")
    res = download.extract_years(start, stop)

    assert len(res) == 1
    assert res[0]["year"] == [2020]

    start = pd.Period("2020-02", freq="M")
    stop = pd.Period("2020-12", freq="M")
    res = download.extract_years(start, stop)

    assert len(res) == 0

    start = pd.Period("2020-01", freq="M")
    stop = pd.Period("2020-11", freq="M")
    res = download.extract_years(start, stop)

    assert len(res) == 0


@pytest.mark.parametrize(
    "start,stop,length",
    [
        ("2020-02", "2020-11", 1),
        ("2020-01", "2020-12", 1),
        ("2020-01", "2022-12", 1),
        ("2020-02", "2022-12", 2),
        ("2020-01", "2022-11", 2),
        ("2020-02", "2022-11", 3),
    ],
)
def test_update_request(start: str, stop: str, length: int) -> None:
    requests = download.update_request_date({}, start, stop)
    assert len(requests) == length


@pytest.mark.parametrize("switch_month_day,length", [(0, 1), (32, 0)])
def test_update_request_no_stop(switch_month_day: int, length: int) -> None:
    prev_month = (
        datetime.date.today().replace(day=1) - datetime.timedelta(days=1)
    ).strftime("%Y-%m")
    requests = download.update_request_date(
        {},
        start=prev_month,
        stop=None,
        switch_month_day=switch_month_day,
    )
    assert len(requests) == length


@pytest.mark.parametrize(
    "stringify_dates,expected_month,expected_day",
    [
        (False, [1], list(range(1, 32))),
        (True, ["01"], [f"{i:02d}" for i in range(1, 32)]),
    ],
)
def test_stringify_dates(
    stringify_dates: bool,
    expected_month: list[str | int],
    expected_day: list[str | int],
) -> None:
    request, *_ = download.update_request_date(
        {}, "2022-1", "2022-1", stringify_dates=stringify_dates
    )
    assert request["month"] == expected_month
    assert request["day"] == expected_day


def test_ensure_request_gets_cached() -> None:
    request = {"f": [2, 1], "e": [1], "d": 1, "c": ["b", "a"], "b": ["ba"], "a": "ba"}
    expected = {"a": "ba", "b": "ba", "c": ["b", "a"], "d": 1, "e": 1, "f": [2, 1]}
    assert download.ensure_request_gets_cached(request) == expected


@pytest.mark.parametrize(
    "chunks, dask_chunks",
    [
        (
            {"time": 1},
            {"forecast_reference_time": (1, 1), "latitude": (2,), "longitude": (2,)},
        ),
        (
            {},
            {"forecast_reference_time": (2,), "latitude": (2,), "longitude": (2,)},
        ),
    ],
)
def test_download_no_transform(
    monkeypatch: pytest.MonkeyPatch,
    chunks: dict[str, int],
    dask_chunks: dict[str, tuple[int, ...]],
) -> None:
    monkeypatch.setattr(cads_toolbox.catalogue, "_download", mock_download)

    ds = download.download_and_transform(*AIR_TEMPERATURE_REQUEST, chunks=chunks)
    assert dict(ds.chunks) == dask_chunks


@pytest.mark.parametrize(
    "transform_chunks, dask_chunks",
    [
        (True, {"forecast_reference_time": (1, 1)}),
        (False, {"forecast_reference_time": (2,)}),
    ],
)
def test_download_and_transform(
    monkeypatch: pytest.MonkeyPatch,
    transform_chunks: bool,
    dask_chunks: dict[str, tuple[int, ...]],
) -> None:
    monkeypatch.setattr(cads_toolbox.catalogue, "_download", mock_download)

    def transform_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.round().mean(("longitude", "latitude"))

    ds = download.download_and_transform(
        *AIR_TEMPERATURE_REQUEST,
        chunks={"time": 1},
        transform_chunks=transform_chunks,
        transform_func=transform_func,
    )
    assert dict(ds.chunks) == dask_chunks
    assert ds["t2m"].values.tolist() == [281.75, 280.75]


@pytest.mark.parametrize("transform_chunks", [True, False])
@pytest.mark.parametrize("invalidate_cache", [True, False])
def test_invalidate_cache(
    monkeypatch: pytest.MonkeyPatch, transform_chunks: bool, invalidate_cache: bool
) -> None:
    monkeypatch.setattr(cads_toolbox.catalogue, "_download", mock_download)

    def transform_func(ds: xr.Dataset) -> xr.Dataset:
        return ds * 0

    ds0 = download.download_and_transform(
        *AIR_TEMPERATURE_REQUEST,
        chunks={"time": 1},
        transform_chunks=transform_chunks,
        transform_func=transform_func,
    )

    def transform_func(ds: xr.Dataset) -> xr.Dataset:  # type: ignore[no-redef]
        return ds * 1

    ds1 = download.download_and_transform(
        *AIR_TEMPERATURE_REQUEST,
        chunks={"time": 1},
        transform_chunks=transform_chunks,
        transform_func=transform_func,
        invalidate_cache=invalidate_cache,
    )

    assert ds0.identical(ds1) is not invalidate_cache
