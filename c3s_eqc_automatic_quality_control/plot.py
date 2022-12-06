"""C3S EQC Automatic Quality Control.

This module offers plot functions to visualise diagnostic results.
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
from typing import Any

import plotly.express as px
import xarray as xr

VAR_NAMES_MAP = {
    "2m_temperature": "t2m",
    "skin_temperature": "skt",
}


def line_plot(
    ds: xr.Dataset,
    var: str,
    title: str = "",
) -> Any:
    try:
        fig = px.line(
            x=ds["time"],
            y=ds[VAR_NAMES_MAP[var]],
        )
    except KeyError as exc:
        raise ValueError(
            f"{var} not available for plot. "
            f"Available variables: {list(VAR_NAMES_MAP.keys())}"
        ) from exc
    fig.update_layout(xaxis_title="time", yaxis_title=var, title=title)
    return fig
