import pathlib
import tempfile
from collections.abc import Generator
from typing import Any

import cacholote
import cdsapi
import fsspec
import pytest
import xarray as xr


class MockResult:
    def __init__(self, name: str, request: dict[str, Any]) -> None:
        self.name = name
        self.request = request

    @property
    def location(self) -> str:
        return tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name

    def download(self, target: str | pathlib.Path | None = None) -> str | pathlib.Path:
        ds = xr.tutorial.open_dataset(self.name).sel(**self.request)
        ds.to_netcdf(path := target or self.location)
        return path


def mock_retrieve(
    self: cdsapi.Client,
    name: str,
    request: dict[str, Any],
    target: str | pathlib.Path | None = None,
) -> fsspec.spec.AbstractBufferedFile:
    result = MockResult(name, request)
    if target is None:
        return result
    return result.download(target)


@pytest.fixture(autouse=True)
def mock_download(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CDSAPI_URL", "")
    monkeypatch.setenv("CDSAPI_KEY", "123456:1123e4567-e89b-12d3-a456-42665544000")
    monkeypatch.setattr(cdsapi.Client, "retrieve", mock_retrieve)


@pytest.fixture(autouse=True)
def set_cache(
    tmpdir: pathlib.Path,
) -> Generator[None, None, None]:
    with cacholote.config.set(
        cache_db_urlpath="sqlite:///" + str(tmpdir / "cacholote.db"),
        cache_files_urlpath=str(tmpdir / "cache_files"),
    ):
        yield
