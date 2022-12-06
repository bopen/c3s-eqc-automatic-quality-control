"""C3S EQC Automatic Quality Control.

This module offers available APIs.
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
from inspect import getmembers, isfunction
import logging
import os
import pathlib
import re
from typing import Dict
import yaml

import cacholote

from . import diagnostics
from . import download
from . import plot


CACHOLOTE_CONFIGS = {
    "cache_files_urlpath": os.getenv("CACHOLOTE_CACHE_FILES_URLPATH", ""),
    "io_delete_original": os.getenv(
        "CACHOLOTE_IO_DELETE_ORIGINAL", "True").lower() in ("true", "1", "on")
}
CATALOG_ALLOWED_KEYS = (
    'product_type',
    'format',
    'time',
    'variable',
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


def show_config_template():
    """Show template configuration file."""
    print(f"{TEMPLATE}")


def list_diagnostics():
    """Return available diagnostic function names."""
    return [f[0] for f in getmembers(diagnostics, isfunction)]


def process_request(
    request: Dict,
):
    day = request.get("switch_month_day")
    if day is None:
        logging.warning(f"No switch month day defined: Default is {SWITCH_MONTH_DAY}")
        day = SWITCH_MONTH_DAY
    reduced = {k: v for k, v in request.items() if k in CATALOG_ALLOWED_KEYS}
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
        cads_request.update({
            var: download.update_request_date(
                reduced, start=request["start"],
                stop=request.get("stop"),
                switch_month_day=day
            )}
        )
    return request, cads_request


def get_next_run_number(
    qar_folder_path: str,
):
    subdirs = list(os.walk(qar_folder_path))
    if not subdirs:
        return 1
    naming_convention = re.compile("run_(?P<n>[0-9]+)$")
    runs = [0]
    for d in subdirs[0][1]:  # list of folders in qar_folder_path
        match = naming_convention.match(d)
        if match is not None:
            runs.append(int(match.group("n")))
    return max(runs) + 1


def prepare_run_workdir(request: Dict, target_dir: str):
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


def run(
    config_file: str,
    target_dir: str,
):
    with open(config_file, "r", encoding="utf-8") as f:
        request = yaml.safe_load(f)

    original_cwd = os.getcwd()
    # Move into qar subfolder
    run_sub, qar_id, run_n = prepare_run_workdir(request, target_dir)
    os.chdir(run_sub)

    request, cads_request = process_request(request)
    chunks = request.get("chunks", {"year": 1, "month": 1})

    for var, req in cads_request.items():
        logging.info(f"Collecting variable '{var}'")
        with cacholote.config.set(**CACHOLOTE_CONFIGS):
            data = download.download_and_transform(
                collection_id=request["collection_id"],
                requests=req,
                chunks=chunks
            )

        # TODO: SANITIZE ATTRS BEFORE SAVING
        logging.info(f"Saving metadata for variable '{var}'")
        with open(run_sub / f"{var}_metadata.yml" , "w", encoding="utf-8") as f:
            f.write(yaml.dump(data.attrs))

        for d in request.get("diagnostics"):
            logging.info(f"Processing diagnostic '{d}' for variable '{var}'")
            diag_ds = getattr(diagnostics, d)(data)

            res = run_sub / f"{var}_{d}.png"
            logging.info(f"Saving diagnostic '{res}'")
            fig = plot.line_plot(diag_ds.squeeze(), var=var)
            fig.write_image(res)

    # Move back into original folder
    os.chdir(original_cwd)
    logging.info(f"QAR ID: {qar_id} - RUN n.: {run_n} finished")
    return
