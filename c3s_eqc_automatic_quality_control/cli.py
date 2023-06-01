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

import os.path

import rich
import typer

from . import cim, dashboard, runner, __version__

STATUSES = {
    "DONE": "[green]DONE[/]",
    "FAILED": "[red]FAILED[/]",
    "PROCESSING": "PROCESSING",
}

app = typer.Typer(
    no_args_is_help=True,
    help="C3S EQC Automatic Quality Control CLI",
)


def _test_workdir(workdir):
    if not os.path.exists(workdir):
        return f"[red]{workdir}[/]"
    return workdir


def eqc() -> None:
    app()


@app.command(name="diagnostics")
def show_diagnostics() -> None:
    """Show available diagnostics names."""
    table = rich.table.Table("Available diagnostics")
    for d in runner.list_diagnostics():
        table.add_row(d)
    rich.print(table)


@app.command(no_args_is_help=True)
def run(
    notebook_path: str,
    qar_id: str = typer.Option("000", "--qar-id", "-q"),
    run_n: str = typer.Option("0", "--run-n", "-n"),
    target_dir: str = typer.Option(..., "--target-dir", "-t"),
) -> None:
    """Run notebook."""
    runner.run(
        notebook_path=notebook_path, qar_id=qar_id, run_n=run_n, target_dir=target_dir
    )


@app.command(name="dashboard")
def run_dashboard(
    qar_id: str = typer.Option(None, "--qar-id", "-q"),
    status: str = typer.Option(None, "--status", "-s"),
    limit: int = typer.Option(20, "--limit", "-l"),
) -> None:
    """Show status of launched processes."""
    table = rich.table.Table("QAR ID", "RUN N.", "STATUS", "START", "STOP")
    table.add_column("WORKDIR", overflow="fold")
    for (qar, run_n), info in dashboard.list_qars(qar_id, status, limit).items():
        table.add_row(
            qar,
            run_n,
            STATUSES[info["status"]],
            info["start"],
            info["stop"],
            _test_workdir(info["workdir"]),
        )
    rich.print(table)


@app.command(name="list-tasks", no_args_is_help=True)
def list_task(
    base_url: str = typer.Option(cim.CIM, "--base-url", "-b"),
    api_user: str = typer.Option(None, "--user", "-u"),
    api_passwd: str = typer.Option(None, "--pass", "-p"),
) -> None:
    """List QAR tasks."""
    if None in (api_user, api_passwd):
        api_user, api_passwd = cim.get_api_credentials()
    res = cim.get_tasks(base_url, user=api_user, passwd=api_passwd)
    rich.print(res)


@app.command(name="push-notebook", no_args_is_help=True)
def push(
    notebook_path: str,
    repo_url: str = typer.Option(None, "--repo", "-r"),
    branch: str = typer.Option("notebooks", "--branch", "-b"),
    user_dir: str = typer.Option("user", "--user", "-u"),
) -> None:
    """Push rendered notebooks."""
    cim.push_notebook(
        notebook_path=notebook_path,
        repo_url=repo_url,
        branch=branch,
        user_dir=user_dir,
    )


@app.command(name="info")
def info() -> None:
    """Print info about EQC AQC installation."""
    etc = dashboard.ensure_log_dir()
    rich.print(f"EQC AQC version: {__version__}")
    rich.print("LOG DIR:", etc)


if __name__ == "__main__":
    eqc()
