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

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objs as go
import xarray as xr
from cartopy.mpl.geocollection import GeoQuadMesh
from xarray.plot.facetgrid import FacetGrid

from . import diagnostics

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
    vars: str | list[str],
    ds_mean: xr.Dataset,
    ds_std: xr.Dataset | None = None,
    hue_dim: str | None = None,
    title: str | None = None,
    x_dim: str = "time",
) -> go.Figure:

    if isinstance(vars, str):
        vars = [vars]
    if hue_dim:
        colors = px.colors.sample_colorscale(
            "turbo", ds_mean.sizes[hue_dim], colortype="tuple"
        )
    elif len(vars) > 10:
        colors = px.colors.sample_colorscale("turbo", len(vars), colortype="tuple")
    else:
        colors = px.colors.qualitative.Plotly
    colors = iter(colors)

    if hue_dim:
        _, means = zip(*ds_mean.groupby(hue_dim))
        if ds_std:
            _, stds = zip(*ds_std.groupby(hue_dim))
        else:
            stds = tuple(xr.Dataset() for _ in range(len(means)))
    else:
        means = (ds_mean,)
        if ds_std:
            stds = (ds_std,)
        else:
            stds = tuple(xr.Dataset() for _ in range(len(means)))

    data = []
    for mean, std in zip(means, stds):
        for var in vars:
            rgb = next(colors)
            if not isinstance(rgb, tuple):
                rgb = pc.hex_to_rgb(rgb)

            da_mean = mean[var].where(mean[var].notnull(), drop=True).squeeze()
            if da_mean.size <= 1:
                continue

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
                    x=da_mean[x_dim],
                    y=da_mean,
                    mode="lines",
                    line=dict(color=dark),
                )
            )
            if std:
                da_std = std[var].where(mean[var].notnull(), drop=True).squeeze()
                data.append(
                    go.Scatter(
                        name="Upper Bound",
                        x=da_mean[x_dim],
                        y=da_mean + da_std,
                        mode="lines",
                        line=dict(width=0.25, color=dark),
                        showlegend=False,
                    )
                )
                data.append(
                    go.Scatter(
                        name="Lower Bound",
                        x=da_mean[x_dim],
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
        legend=dict(yanchor="top", y=1, xanchor="left", x=1),
    )

    return fig


levels = range(-30, 31, 5)
colormap = "YlOrRd"


def projected_map(
    da: xr.DataArray, projection: ccrs.Projection = ccrs.Robinson(), **kwargs: Any
) -> GeoQuadMesh | FacetGrid[Any]:
    """Plot projected map.

    Parameters
    ----------
    da: DataArray
        DataArray to plot
    projection: ccrs.Projection
        Projection for the plot
    **kwargs:
        Keyword arguments for `da.plot`

    Returns
    -------
    GeoQuadMesh or FacetGrid
    """
    # Set defaults
    subplot_kws = kwargs.setdefault("subplot_kws", dict())
    subplot_kws.setdefault("projection", projection)
    kwargs.setdefault("transform", ccrs.PlateCarree())

    # Plot
    p = da.plot(**kwargs)

    # Add coastlines and gridlines
    if isinstance(p, FacetGrid):
        for ax in p.axs.flat:
            ax.coastlines()
            ax.gridlines()
    else:
        p.axes.coastlines()
        p.axes.gridlines(draw_labels=True)

        # Compute statistics
        dataarrays = [diagnostics.spatial_weighted_statistics(da)]
        for stat in "min", "max":
            dataarrays.append(getattr(da, stat)().expand_dims(statistic=[stat]))
        da_stats = xr.merge(dataarrays)[da.name]

        # Add statistics box
        txt = "\n".join(
            [
                f"{k:>10}: {v.squeeze().values:f} {da.attrs.get('units', '')}"
                for k, v in da_stats.groupby("statistic")
            ]
        )
        plt.figtext(
            1, 0.5, txt, ha="left", va="center", figure=p.figure, fontfamily="monospace"
        )

    return p
