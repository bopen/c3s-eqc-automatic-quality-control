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
import re
from typing import Dict

import yaml

from . import diagnostics
from . import download
from . import plot


CATALOG_ALLOWED_KEYS = (
    'product_type',
    'format',
    'time',
    'variable',
)
SWITCH_MONTH_DAY = 9
TEMPLATE = """
qar_id: qar_id
collection_id: reanalysis-era5-single-levels
product_type:  reanalysis
format: grib
time: [06, 18]
variable: 2m_temperature
start: 2021-06
stop: 2021-07
diagnostics:
  - spatial_daily_mean
chunks:
  year: 1
  month: 1
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
        logging.info(f"No switch month day defined: Default is {SWITCH_MONTH_DAY}")
        day = SWITCH_MONTH_DAY
    reduced = {k: v for k, v in request.items() if k in CATALOG_ALLOWED_KEYS}
    cads_request = download.update_request_date(
        reduced, start=request["start"],
        stop=request.get("stop"),
        switch_month_day=day
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


def get_qar_params(
    target_dir: str,
    qar_id: str
):
    qar_folder_path = os.path.join(target_dir, qar_id)
    qar_run_number = get_next_run_number(qar_folder_path)
    logging.info(f"Processing QAR ID: {qar_id} - RUN n.: {qar_run_number}")
    return qar_folder_path, qar_run_number


def run(
    config_file: str,
    target_dir: str,
):
    original_cwd = os.getcwd()
    with open(config_file, "r", encoding="utf-8") as f:
        request = yaml.safe_load(f)

    # Move into qar subfolder
    qar_folder_path, qar_run_number = get_qar_params(target_dir, request["qar_id"])
    run_sub = os.path.join(qar_folder_path, f"run_{qar_run_number}")
    os.makedirs(run_sub)
    os.chdir(run_sub)

    request, cads_request = process_request(request)
    chunks = request.get("chunks", {"year": 1, "month": 1})

    data = download.download_and_transform(
        collection_id=request["collection_id"],
        requests=cads_request,
        chunks=chunks
    )

    # TODO: SANITIZE ATTRS BEFORE SAVING
    with open(os.path.join(run_sub, "meta.yml") , "w", encoding="utf-8") as f:
        f.write(yaml.dump(data.attrs))

    for d in request.get("diagnostics"):
        if d in list_diagnostics():
            fig = plot.line_plot(
                getattr(diagnostics, d)(data).squeeze(),
                var=request["variable"]
            )
            d_res = f"{os.path.join(run_sub, d)}.png"
            logging.info(f"Saving result for : {d_res}")
            fig.write_image(d_res)
        else:
            logging.warn(
                f"Skipping diagnostic {d} since is not available. "
                "Run 'eqc diagnostics' to see available diagnostics."
            )

    # Move back into original folder
    os.chdir(original_cwd)
    return
