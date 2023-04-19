import pathlib
import tempfile
from typing import Any

import fsspec
import xarray as xr


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
