import pathlib
import tempfile
from typing import Any

import cads_toolbox
import fsspec
import pandas as pd
import pytest
import xarray as xr

from c3s_eqc_automatic_quality_control import download


def mock_download(
    collection_id: str,
    request: dict[str, Any],
    target: str | pathlib.Path | None = None,
) -> fsspec.spec.AbstractBufferedFile:
    ds = xr.tutorial.open_dataset(collection_id).sel(**request)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        filename = f.name
    ds.to_netcdf(filename)
    with fsspec.open(filename, "rb") as f:
        return f


def test_split_request() -> None:
    request = {
        "product_type": "reanalysis",
        "format": "grib",
        "variable": "temperature",
        "pressure_level": [
            "1",
            "2",
            "3",
        ],
        "year": [
            "2019",
            "2020",
            "2021",
            "2022",
        ],
        "month": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
        ],
        "day": "01",
        "time": "00:00",
    }

    requests = download.split_request(request, {"month": 1})
    assert len(requests) == 12

    requests = download.split_request(request, {"month": 1, "year": 1})
    assert len(requests) == 4 * 12

    requests = download.split_request(request, split_all=True)
    assert len(requests) == 3 * 4 * 12

    with pytest.raises(
        ValueError, match="`chunks` and `split_all` are mutually exclusive"
    ):
        download.split_request(request, {"month": 1}, split_all=True)


def test_build_chunks() -> None:
    values = list(range(11))

    res = download.build_chunks(values, 1)
    assert res == values

    res = download.build_chunks(values, 11)
    assert res == [values]

    res = download.build_chunks(values, 3)
    assert res[-1] == [9, 10]


def test_check_non_empty_date() -> None:
    request = {
        "year": "2021",
        "month": ["01", "02", "03"],
        "day": ["30", "31"],
    }

    assert download.check_non_empty_date(request)

    request = {
        "year": "2021",
        "month": "03",
        "day": "30",
    }

    assert download.check_non_empty_date(request)

    request = {
        "year": "2021",
        "month": ["02", "04"],
        "day": "31",
    }

    assert not download.check_non_empty_date(request)

    request = {
        "year": "2021",
        "month": "02",
        "day": ["30", "31"],
    }

    assert not download.check_non_empty_date(request)


def test_floor_to_month() -> None:
    date = pd.Period("2022-12", freq="M")
    res = download.floor_to_month(date, 1)

    assert res == pd.Period("2022-01", freq="M")

    date = pd.Period("2022-01", freq="M")
    res = download.floor_to_month(date, month=1)

    assert res == date

    date = pd.Period("2022-12", freq="M")
    res = download.floor_to_month(date, month=12)

    assert res == date

    date = pd.Period("2022-01", freq="M")
    res = download.floor_to_month(date, month=12)

    assert res == pd.Period("2021-12", freq="M")


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


def test_update_request() -> None:
    requests = download.update_request_date({}, "2020-02", "2020-11")
    assert len(requests) == 1

    requests = download.update_request_date({}, "2020-01", "2020-12")
    assert len(requests) == 1

    requests = download.update_request_date({}, "2020-01", "2022-12")
    assert len(requests) == 1

    requests = download.update_request_date({}, "2020-02", "2022-12")
    assert len(requests) == 2

    requests = download.update_request_date({}, "2020-01", "2022-11")
    assert len(requests) == 2

    requests = download.update_request_date({}, "2020-02", "2022-11")
    assert len(requests) == 3


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
        ({"time": 1}, {"time": (1, 1), "latitude": (2,), "longitude": (2,)}),
        ({}, {"time": (2,), "latitude": (2,), "longitude": (2,)}),
    ],
)
def test_donwload_no_transform(
    monkeypatch: pytest.MonkeyPatch,
    chunks: dict[str, int],
    dask_chunks: dict[str, tuple[int, ...]],
) -> None:
    monkeypatch.setattr(cads_toolbox.catalogue, "_download", mock_download)

    ds = download.download_and_transform(
        collection_id="air_temperature",
        requests={
            "time": ["2013-01-01T00", "2013-01-02T00"],
            "lat": [75.0, 72.5],
            "lon": [200.0, 202.5],
        },
        chunks=chunks,
    )
    assert dict(ds.chunks) == dask_chunks


@pytest.mark.parametrize(
    "transform_chunks, dask_chunks",
    [
        (True, {"time": (1, 1)}),
        (False, {"time": (2,)}),
    ],
)
def test_donwload_and_transform(
    monkeypatch: pytest.MonkeyPatch,
    transform_chunks: bool,
    dask_chunks: dict[str, tuple[int, ...]],
) -> None:
    monkeypatch.setattr(cads_toolbox.catalogue, "_download", mock_download)

    def transform_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.mean(("longitude", "latitude")).round()

    ds = download.download_and_transform(
        collection_id="air_temperature",
        requests={
            "time": ["2013-01-01T00", "2013-01-02T00"],
            "lat": [75.0, 72.5],
            "lon": [200.0, 202.5],
        },
        chunks={"time": 1},
        transform_chunks=transform_chunks,
        transform_func=transform_func,
    )
    assert dict(ds.chunks) == dask_chunks
    assert ds["air"].values.tolist() == [243, 244]
