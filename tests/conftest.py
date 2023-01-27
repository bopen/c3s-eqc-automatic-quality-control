import pathlib
from typing import Iterator

import cacholote
import pytest


@pytest.fixture(autouse=True)
def config_cacholote(tmpdir: pathlib.Path) -> Iterator[None]:
    with cacholote.config.set(
        cache_db_urlpath="sqlite:///" + str(tmpdir / "cacholote.db"),
        cache_files_urlpath=str(tmpdir / "cache_files"),
    ):
        yield
