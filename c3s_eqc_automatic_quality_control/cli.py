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

import rich
import rich.logging
import typer

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

app = typer.Typer(
    no_args_is_help=True,
    help="C3S EQC Automatic Quality Control CLI",
)


def eqc():
    app()


@app.command(name="diagnostics")
def show_diagnostics():
    """Show available diagnostic function names."""
    table = rich.table.Table("Available diagnostics")
    for d in api.list_diagnostics():
        table.add_row(d)
    rich.print(table)


@app.command(name="show-config-template")
def show_config_template():
    """Show template configuration file."""
    rich.print(api.TEMPLATE)


@app.command()
def run(
    config_file: str,
    target_dir: str = typer.Option(os.getcwd(), "--target-dir", "-t"),
):
    api.run(config_file, target_dir)


if __name__ == "__main__":
    eqc()
