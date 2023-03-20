"""C3S EQC Automatic Quality Control.

This module offers available APIs.
"""

import logging
import os
import pathlib
import uuid

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
from inspect import getmembers, isfunction

import nbclient
import nbformat

from . import dashboard, diagnostics

LOGGER = dashboard.get_logger()


def list_diagnostics() -> list[str]:
    """Return available diagnostic function names."""
    return [
        f[0] for f in getmembers(diagnostics, isfunction) if not f[0].startswith("_")
    ]


def _prepare_run_workdir(
    target_dir: str,
    qar_id: str,
    run_n: str,
    logger: logging.Logger = LOGGER,
) -> pathlib.Path:
    run_sub = pathlib.Path(target_dir).resolve() / f"qar_{qar_id}" / f"run_{run_n}"
    logger.info(f"QAR workdir: {run_sub}")
    try:
        os.makedirs(run_sub)
    except FileExistsError:
        logger.warning(
            f"Run '{run_n}' for qar '{qar_id}' already exists. "
            "Results will be overwritten."
        )
    return run_sub


def run(
    notebook_path: str,
    qar_id: str,
    run_n: str,
    target_dir: str,
) -> None:
    run_id = str(uuid.uuid4())[:8]
    msg = f"QAR ID: {qar_id} - RUN n.: {run_n}"

    logger = dashboard.get_eqc_run_logger(f"qar_{qar_id}_run_{run_n}_{run_id}")
    logger.info(f"{msg} - PROCESSING")

    notebook_resolved = pathlib.Path(notebook_path).resolve()
    executed_notebook = os.path.basename(notebook_resolved)
    run_sub = _prepare_run_workdir(target_dir, qar_id, run_n, logger)
    original_cwd = os.getcwd()
    # Move into qar subfolder
    os.chdir(run_sub)
    try:
        nb = nbformat.read(
            notebook_resolved, as_version=4
        )  # type: ignore[no-untyped-call]
        client = nbclient.NotebookClient(nb)  # type: ignore[attr-defined]
        client.execute()
        nbformat.write(nb, executed_notebook)  # type: ignore[no-untyped-call]
        logger.info(f"Rendered notebook: {run_sub}/{executed_notebook}")
    except:  # noqa: E722
        logger.exception(f"{msg} - FAILED")
    else:
        logger.info(f"{msg} - DONE")
    finally:
        # Move back into original folder
        os.chdir(original_cwd)
