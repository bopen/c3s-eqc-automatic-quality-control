"""C3S EQC Automatic Quality Control.

This module manages the command line interfaces.
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
import os

import click
import rich.logging

from . import api


logging.basicConfig(
    level="INFO",
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


@click.group()
@click.version_option(version=None, prog_name="eqc", message="C3S EQC AQC %(version)s")
def eqc():
    """C3S EQC Automatic Quality Control CLI"""
    pass


@click.command()
def show_diagnostics():
    """Show available diagnostic function names."""
    for d in api.list_diagnostics():
        print(d)


eqc.add_command(show_diagnostics, "diagnostics")


@click.command()
def show_config_template():
    """Show template configuration file."""
    api.show_config_template()


eqc.add_command(show_config_template, "config-template")


@click.command()
@click.argument("config_file", type=click.STRING, required=False)
@click.option(
    "target_dir",
    "-t",
    default=os.getcwd(),
    help="Path to the output folder. Default to current directory."
)
def run(
    config_file: str,
    target_dir: str,
):
    """Update Quality Assurance Report."""
    if config_file is None:
        logging.error("QAR config file is required.")
        logging.error("Sample config:")
        api.show_config_template()
        return

    api.run(config_file, target_dir)


eqc.add_command(run, "run")
