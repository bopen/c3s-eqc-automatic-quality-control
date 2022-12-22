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

import rich
import typer

from . import dashboard, runner

STATUSES = {
    "DONE": "[green]DONE[/]",
    "FAILED": "[red]FAILED[/]",
    "PROCESSING": "PROCESSING",
}

app = typer.Typer(
    no_args_is_help=True,
    help="C3S EQC Automatic Quality Control CLI",
)


def eqc() -> None:
    app()


@app.command(name="diagnostics")
def show_diagnostics() -> None:
    """Show available diagnostics names."""
    table = rich.table.Table("Available diagnostics")
    for d in runner.list_diagnostics():
        table.add_row(d)
    rich.print(table)


@app.command(name="show-config-template")
def show_config_template() -> None:
    """Show a template configuration file."""
    rich.print(runner.TEMPLATE)


@app.command()
def run(
    config_file: str,
    target_dir: str = typer.Option(..., "--target-dir", "-t"),
) -> None:
    """Run automatic quality checks and populate QAR."""
    runner.run(config_file, target_dir)


@app.command(name="dashboard")
def dasboard(
    qar_id: str = typer.Option(None, "--qar-id", "-q"),
    status: str = typer.Option(None, "--status", "-s"),
    limit: int = typer.Option(20, "--limit", "-l"),
) -> None:
    """Show status of launched processes."""
    table = rich.table.Table("QAR ID", "RUN N.", "STATUS", "START", "STOP", "WORKDIR")
    for (qar, run_n), info in dashboard.list_qars(qar_id, status, limit).items():
        table.add_row(
            qar,
            run_n,
            STATUSES[info["status"]],
            info["start"],
            info["stop"],
            info["workdir"],
        )
    rich.print(table)


if __name__ == "__main__":
    eqc()
