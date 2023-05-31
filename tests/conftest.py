import pathlib
from collections.abc import Generator

import cacholote
import pytest


@pytest.fixture(autouse=True)
def set_cache(
    tmpdir: pathlib.Path,
) -> Generator[None, None, None]:
    with cacholote.config.set(
        cache_db_urlpath="sqlite:///" + str(tmpdir / "cacholote.db"),
        cache_files_urlpath=str(tmpdir / "cache_files"),
    ):
        yield
