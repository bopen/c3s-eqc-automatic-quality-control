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
API_USER = None
API_PWD = None


def http_request(url: str, request: str = "get", **kwargs: Any) -> Any:
    method = getattr(requests, request)
    res = method(url, **kwargs)
    if res.status_code not in (200, 202):
        raise RuntimeError(f"Error {res.status_code}: {res.reason}")
    return res


def get_api_credentials() -> Tuple[str, str]:
    global API_USER
    global API_PWD
    if API_USER is None and API_PWD is None:
        user = os.environ.get(API_USER_VAR_NAME)
        pwd = os.environ.get(API_PWD_VAR_NAME)
        if user is None or pwd is None:
            raise ValueError(
                "No authentication provided. EQC_AQC_API_USER and EQC_AQC_API_PWD need to be set."
            )
        API_USER, API_PWD = user, pwd
    return API_USER, API_PWD


def get_tasks(baseurl: str, user: str, passwd: str) -> Any:
    endpoint = baseurl + "workflows/cds/tasks"
    res = http_request(url=endpoint, auth=(user, passwd))
    return res.json().get("content")


def push_notebooks(
    notebook_paths: str | list[str], repo_url: str, branch: str, user_dir: str
) -> None:
    if isinstance(notebook_paths, str):
        notebook_paths = [notebook_paths]

    with tempfile.TemporaryDirectory() as tempdir:
        # Clone the repository
        repo = git.Repo.clone_from(repo_url, tempdir, branch=branch)

        dest_dir = f"{tempdir}/rendered_notebooks/{user_dir}"
        try:
            os.makedirs(dest_dir)
        except FileExistsError:
            pass
        executed = []
        for nb in notebook_paths:
            # Move the file to the folder
            shutil.copy(pathlib.Path(nb).resolve(), dest_dir)

            # Add the file
            executed_nb = os.path.basename(nb)
            executed.append(executed_nb)
            repo.git.add(f"{dest_dir}/{executed_nb}")
        commit_message = f"Add notebooks: {', '.join(executed)}"
        # Commit the file
        repo.index.commit(commit_message)

        # Push the changes
        origin = repo.remote(name="origin")
        origin.push(refspec=f"{branch}:{branch}")


def push_qar(workdir: str, repo_url: str, branch: str, user_dir: str) -> None:
    notebook_paths = [
        str(nb.absolute()) for nb in pathlib.Path(workdir).glob("*.ipynb")
    ]
    push_notebooks(
        notebook_paths=notebook_paths,
        repo_url=repo_url,
        branch=branch,
        user_dir=user_dir,
    )
