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
import pathlib
import re
from typing import Any, Dict

import rich.logging

LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_TIME_FMT = "%Y-%m-%d %H:%M:%S"
FILENAME_TIME_FMT = "%Y%m%d%H%M%S"
MSG_REGEX = "(?:.+) - QAR ID: (?P<qar_id>.+) - RUN n.: (?P<run_n>.+) - (?P<status>.+)"
FILENAME_REGEX = "eqc_(?P<start>[0-9]{14})"

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
    log_dir = pathlib.Path.home() / ".eqc/logs/"
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_eqc_run_logger(name: str) -> logging.Logger:
    now = datetime.datetime.now().strftime(FILENAME_TIME_FMT)
    log_dir = ensure_log_dir()
    logger = get_logger(f"{name}")
    logger = set_logfile(logger, log_dir / f"eqc_{now}_{name}.log")
    return logger


def list_qars() -> Dict[Any, Any]:
    log_dir = ensure_log_dir()
    status: Dict[Any, Any] = {}
    for log in log_dir.glob("eqc*.log"):
        with open(log, "r", encoding="utf-8") as f:
            # get only matched lines
            for match in map(re.compile(MSG_REGEX).match, f.readlines()):
                if match is not None:
                    info = match.groupdict()
            # start datetime from filename
            # update status for each qar_id, run_n if run_n is newer
            start = re.compile(FILENAME_REGEX).match(log.name)
            if start is not None:
                s = start.group("start")
                info.update(
                    {
                        "start": datetime.datetime.strptime(
                            s, FILENAME_TIME_FMT
                        ).strftime(LOG_TIME_FMT)
                    }
                )
            qar_id = info.pop("qar_id")
            run_n = info.pop("run_n")
            if (qar_id, run_n) not in status or s > status[(qar_id, run_n)]["start"]:
                status[(qar_id, run_n)] = info
    return status
