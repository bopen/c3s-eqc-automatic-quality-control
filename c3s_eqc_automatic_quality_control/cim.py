"""C3S EQC Automatic Quality Control.

This module offers interfaces with the CIM API.
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

import os
import pathlib
import shutil
import tempfile
from typing import Any, Tuple

import git
import requests

API_USER_VAR_NAME = "EQC_AQC_API_USER"
API_PWD_VAR_NAME = "EQC_AQC_API_PWD"
CIM = "https://cds2-backoffice-dev.copernicus-climate.eu/api/v1/"
RENDERED = "https://github.com/bopen/c3s-eqc-toolbox-template.git"


def http_request(url: str, request: str = "get", **kwargs: Any) -> Any:
    method = getattr(requests, request)
    res = method(url, **kwargs)
    if res.status_code not in (200, 202):
        raise RuntimeError(f"Error {res.status_code}: {res.reason}")
    return res


def get_api_credentials() -> Tuple[str, str]:
    user = os.environ.get(API_USER_VAR_NAME)
    pwd = os.environ.get(API_PWD_VAR_NAME)
    if user is None or pwd is None:
        raise ValueError(
            "No authentication provided. EQC_AQC_API_USER and EQC_AQC_API_PWD need to be set."
        )
    return user, pwd


def get_tasks(baseurl: str, user: str, passwd: str) -> Any:
    endpoint = baseurl + "workflows/cds/tasks"
    res = http_request(url=endpoint, auth=(user, passwd))
    return res.json().get("content")


def get_task(baseurl: str, user: str, passwd: str, task_id: str) -> Any:
    tasks = get_tasks(baseurl, user, passwd)
    for task in tasks:
        if task["id"] == task_id:
            return task
    else:
        raise ValueError(f"No task available with ID: {task_id}")


def push_notebook(
    notebook_path: str, repo_url: str, branch: str, user_dir: str
) -> None:
    if repo_url is None:
        repo_url = RENDERED

    with tempfile.TemporaryDirectory() as tempdir:
        # Clone the repository
        repo = git.Repo.clone_from(repo_url, tempdir, branch=branch)  # type: ignore[attr-defined]

        relative_dest_dir = f"rendered_notebooks/{user_dir}"
        dest_dir = f"{tempdir}/{relative_dest_dir}"
        try:
            os.makedirs(dest_dir)
        except FileExistsError:
            pass
        executed = []
        # Move the file to the folder
        shutil.copy(pathlib.Path(notebook_path).resolve(), dest_dir)

        # Add the file
        executed_nb = os.path.basename(notebook_path)
        executed.append(executed_nb)
        repo.git.add(f"{dest_dir}/{executed_nb}")

        # Commit the file
        commit_message = f"Add notebooks: {', '.join(executed)}"
        commit = repo.index.commit(commit_message)

        # Push the changes
        origin = repo.remote(name="origin")
        origin.push(refspec=f"{branch}:{branch}")

        # Build permalink
        base_url, _ = os.path.splitext(repo_url)
        return f"{base_url}/blob/{commit}/{relative_dest_dir}/{executed_nb}"


def add_qar_url(baseurl: str, user: str, passwd: str, task_id: str, url: str) -> None:
    # Check task_id is a valid task
    get_task(baseurl, user, passwd, task_id)

    payload = {"review_jupyter_url": {"jupyter_notebook_url": url}}
    endpoint = f"{baseurl}/workflows/cds/tasks/{task_id}"
    http_request(url=endpoint, auth=(user, passwd), request="post", json=payload)
