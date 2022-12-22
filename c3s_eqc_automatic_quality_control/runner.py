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
from typing import Any

import yaml

from . import dashboard, diagnostics, download, plot

_CATALOG_ALLOWED_KEYS = (
    "product_type",
    "format",
    "time",
    "variable",
)
LOGGER = dashboard.get_logger()
SWITCH_MONTH_DAY = 9
TEMPLATE = """
qar_id: qar_identifier
run_n: run_number
collection_id: reanalysis-era5-single-levels
product_type:  reanalysis
format: grib
time: [06, 18]
variables:
  - 2m_temperature
  - skin_temperature
start: YYYY-MM
stop: YYYY-MM
diagnostics:
  - spatial_daily_mean
chunks:
  year: 1
  month: 1
switch_month_day: 9
"""


def show_config_template() -> None:
    """Show template configuration file."""
    print(f"{TEMPLATE}")


def list_diagnostics() -> list[str]:
    """Return available diagnostic function names."""
    return [
        f[0] for f in getmembers(diagnostics, isfunction) if not f[0].startswith("_")
    ]


def process_request(
    request: dict[Any, Any],
    logger: logging.Logger = LOGGER,
) -> tuple[dict[Any, Any], list[str]]:
    day = request.get("switch_month_day")
    if day is None:
        day = SWITCH_MONTH_DAY
    reduced = {k: v for k, v in request.items() if k in _CATALOG_ALLOWED_KEYS}
    cads_request = {}
    diagnos = []
    for diagnostic in request["diagnostics"]:
        if diagnostic not in list_diagnostics():
            logger.warning(
                f"Skipping diagnostic '{diagnostic}' since is not available. "
                "Run 'eqc diagnostics' to see available diagnostics."
            )
        else:
            diagnos.append(diagnostic)

    # Request to CADS are single variable only
    for var in request["variables"]:
        reduced["variable"] = var
        cads_request.update(
            {
                var: download.update_request_date(
                    reduced,
                    start=request["start"],
                    stop=request.get("stop"),
                    switch_month_day=day,
                )
            }
        )
    return cads_request, diagnos


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


def run_aqc(
    request: dict[Any, Any],
    logger: logging.Logger = LOGGER,
) -> None:
    cads_request, diagnos = process_request(request)
    chunks = request.get("chunks", {"year": 1, "month": 1})

    for var, req in cads_request.items():
        logger.info(f"Collecting variable '{var}'")
        data = download.download_and_transform(
            collection_id=request["collection_id"],
            requests=req,
            chunks=chunks,
            logger=logger,
        )

        # TODO: SANITIZE ATTRS BEFORE SAVING
        logger.info(f"Saving metadata for variable '{var}'")
        with open(f"{var}_metadata.yml", "w", encoding="utf-8") as f:
            f.write(yaml.dump(data.attrs))

        for d in diagnos:
            logger.info(f"Processing diagnostic '{d}' for variable '{var}'")
            diag_ds = getattr(diagnostics, d)(data)

            res = f"{var}_{d}.png"
            logger.info(f"Saving diagnostic: '{res}'")
            fig = plot.line_plot(diag_ds.squeeze(), var=var, title=d)
            fig.write_image(res)
    return


def run(
    config_file: str,
    target_dir: str,
) -> None:
    with open(config_file, encoding="utf-8") as f:
        request = yaml.safe_load(f)

    qar_id = request.get("qar_id")
    run_n = request.get("run_n", 0)
    run_id = str(uuid.uuid4())[:8]
    msg = f"QAR ID: {qar_id} - RUN n.: {run_n}"

    logger = dashboard.get_eqc_run_logger(f"qar_{qar_id}_run_{run_n}_{run_id}")
    logger.info(f"{msg} - PROCESSING")

    original_cwd = os.getcwd()
    run_sub = _prepare_run_workdir(target_dir, qar_id, run_n, logger)
    # Move into qar subfolder
    os.chdir(run_sub)

    try:
        run_aqc(request, logger)
    except Exception:
        logger.exception(f"{msg} - FAILED")
    else:
        logger.info(f"{msg} - DONE")
    finally:
        # Move back into original folder
        os.chdir(original_cwd)
