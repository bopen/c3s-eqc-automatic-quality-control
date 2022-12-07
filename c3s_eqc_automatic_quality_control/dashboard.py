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

import logging
import pathlib
from typing import Any, Dict
import datetime

import rich.logging


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

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


def set_logfile(logger: logging.Logger, logfilepath: str) -> logging.Logger:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(logfilepath)
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_TIME_FORMAT
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def ensure_log_dir() -> pathlib.Path:
    log_dir = pathlib.Path.home() / ".eqc/logs/"
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_eqc_run_logger(name) -> logging.Logger:
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = ensure_log_dir()
    logger = get_logger(f"{name}")
    logger = set_logfile(logger, log_dir / f"aqc_{now}_{name}.log")
    return logger
