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

import pathlib
import os
import requests
from typing import Any, Tuple
import yaml


API_USER_VAR_NAME = "EQC_AQC_API_USER"
API_PWD_VAR_NAME = "EQC_AQC_API_PWD"
CIM = "https://cds2-backoffice-dev.copernicus-climate.eu/api/v1/"
API_USER = None
API_PWD = None


def http_request(url, request="get", **kwargs):
    method = getattr(requests, request)
    res = method(url, **kwargs)
    if res.status_code not in (200, 202):
        raise RuntimeError(f"Error {res.status_code}: {res.reason}")
    return res


def get_api_credentials() -> list[str, str]:
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


def get_tasks(
    baseurl: str,
    auth: Tuple
) -> list:
    endpoint = baseurl + "workflows/cds/tasks"
    res = http_request(url=endpoint, auth=auth)
    return res.json().get("content")


def push_image_to_task(
    baseurl: str,
    task_id: str,
    image_path: str,
    auth: Tuple
) -> str:
    contents = get_tasks(baseurl, auth)
    if not contents:
        raise ValueError("No open task available.")
    endpoint = baseurl + f"workflows/cds/tasks/{task_id}/attachments"
    with open(image_path, "rb") as f:
        files = {"image": f.read()}
    res = http_request(endpoint, "post", auth=auth, files=files)
    return res.json().get("id")


def push_attrs_to_task(baseurl, task_id, metadata, auth) -> None:
    headers = {
        "Content-Type": "application/json",
    }
    endpoint = baseurl + f"workflows/cds/tasks/{task_id}/images"
    json_data = {"images": metadata}
    return http_request(endpoint, "post", auth=auth, headers=headers, json=json_data)


def push_to_task(baseurl, task_id, workdir, auth):
    im_paths = [str(im.absolute()) for im in pathlib.Path(workdir).glob("*.png")]
    meta_paths = [str(meta.absolute()) for meta in pathlib.Path(workdir).glob("*.yml")]
    if len(im_paths) != len(meta_paths):
        raise ValueError(
            f"n. of images {len(im_paths)} and n. of metadata {len(meta_paths)} do not match."
        )
    image_ids = [push_image_to_task(baseurl, task_id, im, auth) for im in im_paths]
    print(f"image id-s: {image_ids}")
    metadata = []
    for image_id, meta in zip(image_ids, meta_paths):
        with open(meta, "r") as f:
            im_attrs = yaml.safe_load(f.read())
            im_attrs.update({"imageId": image_id})
            metadata.append(im_attrs)
    return push_attrs_to_task(baseurl, task_id, metadata, auth)
