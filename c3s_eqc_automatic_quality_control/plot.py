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

import plotly.colors as pc
import plotly.express as px
import plotly.graph_objs as go
import xarray as xr

VAR_NAMES_MAP = {
    "2m_temperature": "t2m",
    "skin_temperature": "skt",
}


def line_plot(
    ds: xr.Dataset,
    var: str,
    title: str = "",
) -> go.Figure:
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


def shaded_std(
    ds_mean: xr.Dataset,
    ds_std: xr.Dataset,
    vars: list[str],
    hue_dim: str | None = None,
    title: str | None = None,
) -> go.Figure:

    data = []
    colors = iter(px.colors.qualitative.Plotly)
    if isinstance(vars, str):
        vars = [vars]

    if hue_dim:
        _, means = zip(*ds_mean.groupby(hue_dim))
        _, stds = zip(*ds_std.groupby(hue_dim))
    else:
        means = (ds_mean, )
        stds = (ds_std, )

    for mean, std in zip(means, stds):
        for var in vars:
            rgb = next(colors)
            rgb = pc.hex_to_rgb(rgb)

            da_mean = mean[var].squeeze()
            da_std = std[var].squeeze()

            rgb = list(rgb)
            dark = f"rgba{tuple(rgb + [1])}"
            light = f"rgba{tuple(rgb + [.15])}"
            if hue_dim:
                label = str(da_mean[hue_dim].values)
            else:
                label = da_mean.attrs.get("long_name", var)
            data.append(
                go.Scatter(
                    name=label,
                    x=da_mean["time"],
                    y=da_mean,
                    mode="lines",
                    line=dict(color=dark),
                )
            )
            data.append(
                go.Scatter(
                    name="Upper Bound",
                    x=da_mean["time"],
                    y=da_mean + da_std,
                    mode="lines",
                    line=dict(width=0.25, color=dark),
                    showlegend=False,
                )
            )
            data.append(
                go.Scatter(
                    name="Lower Bound",
                    x=da_mean["time"],
                    y=da_mean - da_std,
                    line=dict(width=0.25, color=dark),
                    mode="lines",
                    fillcolor=light,
                    fill="tonexty",
                    showlegend=False,
                )
            )
        fig = go.Figure(data)

        fig.update_layout(
            yaxis_title=ds_mean[var].attrs.get("units", ""),
            title=title,
            hovermode="x",
            legend=dict(yanchor="bottom", y=1, xanchor="right", x=1),
        )
    return fig
