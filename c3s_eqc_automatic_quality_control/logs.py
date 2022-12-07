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

import rich.logging


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


def get_logger(name: str = "aqc", level: int = logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level)
    return logger


def set_logfile(logger: logging.Logger, logfile: str):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(logfile)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
