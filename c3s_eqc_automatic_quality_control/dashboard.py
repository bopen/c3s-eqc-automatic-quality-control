"""C3S EQC Automatic Quality Control.

This module manages the package logging.
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

import datetime
import logging
import os
import pathlib
import re
from operator import itemgetter
from typing import Any

import rich.logging

EQC_AQC_ENV_VARNAME = "EQC_AQC_DIR"
LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_TIME_FMT = "%Y-%m-%d %H:%M:%S"
FILENAME_TIME_FMT = "%Y%m%d%H%M%S"
MSG_REGEX = "(?P<logtime>.+) - (?:.+) - QAR ID: (?P<qar_id>.+) - RUN n.: (?P<run_n>.+) - (?P<status>.+)"
FILENAME_REGEX = (
    "eqc_(?P<start>[0-9]{14})_qar_(?P<qar_id>.+)_run_(?P<run_n>.+)_(?:.+).log"
)

logging.basicConfig(
    format="%(message)s",
    handlers=[
        rich.logging.RichHandler(
            show_time=False,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
    ],
)


def get_logger(name: str = "eqc", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    return logger


def set_logfile(logger: logging.Logger, logfilepath: pathlib.Path) -> logging.Logger:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(logfilepath)
    formatter = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_TIME_FMT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def ensure_log_dir() -> pathlib.Path:
    if os.environ.get(EQC_AQC_ENV_VARNAME) is None:
        eqc_dir = pathlib.Path.home() / ".eqc"
    else:
        eqc_dir = pathlib.Path(os.environ[EQC_AQC_ENV_VARNAME])
    log_dir_path = eqc_dir / "logs/"
    if not log_dir_path.is_dir():
        log_dir_path.mkdir(parents=True, exist_ok=True)
    return log_dir_path


def get_eqc_run_logger(name: str) -> logging.Logger:
    now = datetime.datetime.now().strftime(FILENAME_TIME_FMT)
    log_dir = ensure_log_dir()
    logger = get_logger(f"{name}")
    logger = set_logfile(logger, log_dir / f"eqc_{now}_{name}.log")
    return logger


def get_most_recent_log(info: list[dict[Any, Any]]) -> dict[Any, Any]:
    # return most recent info base on datetime in logfile name
    sorted_info = sorted(info, key=itemgetter("start"))
    # Use ISO format for date and time
    sorted_info[-1].update(
        {
            "start": datetime.datetime.strptime(
                sorted_info[-1]["start"], FILENAME_TIME_FMT
            ).strftime(LOG_TIME_FMT)
        }
    )
    return sorted_info[-1]


def update_from_logfile(logfile: pathlib.Path, info: dict[Any, Any]) -> dict[Any, Any]:
    with open(logfile, encoding="utf-8") as f:
        # get only last matched line
        status = None
        for line in f:
            line = line.strip("\n")
            # Workdir path
            if "QAR workdir: " in line:
                info.update({"workdir": line.rsplit("QAR workdir: ", 1)[-1]})
                continue
            match = re.compile(MSG_REGEX).match(line)
            if match is not None:
                status = match["status"]
                if match["status"] in ("DONE", "FAILED"):
                    info.update({"stop": match["logtime"]})
                    break
        if status is None:
            raise RuntimeError(f"No status found in logfile {logfile}")
        info.update({"status": status})
    return info


def list_qars(
    qar_id: str | None = None, status: str | None = None, limit: int | None = 20
) -> dict[Any, Any]:
    log_dir = ensure_log_dir()
    qar_map: dict[tuple[str, str], list[dict[Any, Any]]] = {}
    search = "eqc*.log"
    # filter by qar_id
    if qar_id is not None:
        search = f"eqc*{qar_id}*.log"

    for log in sorted(log_dir.glob(search)):
        filename_info = re.compile(FILENAME_REGEX).match(log.name)
        if filename_info is None:
            continue
        qar_id = filename_info["qar_id"]
        run_n = filename_info["run_n"]
        info = {"start": filename_info["start"], "logfile": str(log), "stop": ""}
        info = update_from_logfile(log, info)
        qar_map[(qar_id, run_n)] = qar_map.get((qar_id, run_n), []) + [info]

    latest_status = {k: get_most_recent_log(v) for k, v in qar_map.items()}

    # filter first {limit} results
    latest_status = dict(sorted(latest_status.items(), key=itemgetter(0))[:limit])

    # Filter by status
    if status is not None:
        latest_status = {
            k: v for k, v in latest_status.items() if v["status"] == status.upper()
        }
    return latest_status
