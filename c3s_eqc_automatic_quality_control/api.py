"""C3S EQC Automatic Quality Control.

This module offers available APIs.
"""

import logging
import os
import pathlib

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
from typing import Any, Dict, List, Tuple

import yaml

from . import diagnostics, download, plot

_CATALOG_ALLOWED_KEYS = (
    "product_type",
    "format",
    "time",
    "variable",
)
SWITCH_MONTH_DAY = 9
TEMPLATE = """
qar_id: qar_id
run_n: 0
collection_id: reanalysis-era5-single-levels
product_type:  reanalysis
format: grib
time: [06, 18]
variables:
  - 2m_temperature
  - skin_temperature
start: 2021-06
stop: 2021-07
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


def list_diagnostics() -> List[str]:
    """Return available diagnostic function names."""
    return [f[0] for f in getmembers(diagnostics, isfunction)]


def process_request(
    request: Dict[Any, Any],
) -> Dict[Any, Any]:
    day = request.get("switch_month_day")
    if day is None:
        logging.info(f"No switch month day defined: Default is {SWITCH_MONTH_DAY}")
        day = SWITCH_MONTH_DAY
    reduced = {k: v for k, v in request.items() if k in _CATALOG_ALLOWED_KEYS}
    cads_request = {}
    for d in request["diagnostics"]:
        if d not in list_diagnostics():
            request["diagnostics"].remove(d)
            logging.warning(
                f"Skipping diagnostic '{d}' since is not available. "
                "Run 'eqc diagnostics' to see available diagnostics."
            )

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
    return cads_request


def _prepare_run_workdir(
    request: Dict[Any, Any], target_dir: str
) -> Tuple[pathlib.Path, str, int]:
    qar_id = request.pop("qar_id")
    run_n = request.pop("run_n", 0)
    run_sub = pathlib.Path(target_dir) / qar_id / f"run_{run_n}"
    logging.info(f"Processing QAR ID: {qar_id} - RUN n.: {run_n}")
    try:
        os.makedirs(run_sub)
    except FileExistsError:
        logging.warning(
            f"Run '{run_n}' for qar '{qar_id}' already exists. "
            "Results will be overwritten."
        )
    return run_sub, qar_id, run_n


def run_aqc(
    request: Dict[Any, Any],
) -> None:
    cads_request = process_request(request)
    chunks = request.get("chunks", {"year": 1, "month": 1})

    for var, req in cads_request.items():
        logging.info(f"Collecting variable '{var}'")
        data = download.download_and_transform(
            collection_id=request["collection_id"], requests=req, chunks=chunks
        )

        # TODO: SANITIZE ATTRS BEFORE SAVING
        logging.info(f"Saving metadata for variable '{var}'")
        with open(f"{var}_metadata.yml", "w", encoding="utf-8") as f:
            f.write(yaml.dump(data.attrs))

        for d in request.get("diagnostics", []):
            logging.info(f"Processing diagnostic '{d}' for variable '{var}'")
            diag_ds = getattr(diagnostics, d)(data)

            res = f"{var}_{d}.png"
            logging.info(f"Saving diagnostic: '{res}'")
            fig = plot.line_plot(diag_ds.squeeze(), var=var, title=d)
            fig.write_image(res)
    return


def run(
    config_file: str,
    target_dir: str,
) -> None:
    with open(config_file, "r", encoding="utf-8") as f:
        request = yaml.safe_load(f)
    original_cwd = os.getcwd()

    # Move into qar subfolder
    run_sub, qar_id, run_n = _prepare_run_workdir(request, target_dir)
    os.chdir(run_sub)

    try:
        run_aqc(request)
    except Exception:
        logging.error(f"QAR ID: {qar_id} - RUN n.: {run_n} failed ")
    else:
        logging.info(f"QAR ID: {qar_id} - RUN n.: {run_n} finished")
    finally:
        # Move back into original folder
        os.chdir(original_cwd)
