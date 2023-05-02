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
import warnings
from collections.abc import Hashable, Iterable
from typing import Any

import cartopy.crs as ccrs
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objs as go
import xarray as xr
from cartopy.mpl.geocollection import GeoQuadMesh
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from xarray.plot.facetgrid import FacetGrid

from . import diagnostics, utils

VAR_NAMES_MAP = {
    "2m_temperature": "t2m",
    "skin_temperature": "skt",
}

FLAGS_T = int | Iterable[int]
COLOR_T = str | Iterable[str]


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

            da_mean = (
                mean[var].where(mean[var].notnull().compute(), drop=True).squeeze()
            )
            if da_mean.size <= 1:
                continue

            rgb = list(rgb)
            dark = f"rgba{tuple(rgb + [1])}"
            light = f"rgba{tuple(rgb + [.15])}"

            if hue_dim:
                label = str(da_mean[hue_dim].values)
            else:
                label = xr.plot.utils.label_from_attrs(da_mean)

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
                da_std = (
                    std[var].where(mean[var].notnull().compute(), drop=True).squeeze()
                )
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
    da: xr.DataArray,
    projection: ccrs.Projection = ccrs.PlateCarree(),
    show_stats: bool | None = None,
    plot_func: str | None = None,
    **kwargs: Any,
) -> GeoQuadMesh | FacetGrid[Any]:
    """Plot projected map.

    Parameters
    ----------
    da: DataArray
        DataArray to plot
    projection: ccrs.Projection
        Projection for the plot
    show_stats: bool, optional
        Whether to show or not a box with statistics
    plot_func: str, optional
        Plotting function (e.g., pcolormesh, contourf, ...)
    **kwargs:
        Keyword arguments for `xr.plot`

    Returns
    -------
    GeoQuadMesh or FacetGrid
    """
    # Set defaults
    kwargs.setdefault("transform", ccrs.PlateCarree())
    if "ax" not in kwargs:
        subplot_kws = kwargs.setdefault("subplot_kws", dict())
        subplot_kws.setdefault("projection", projection)

    # Plot
    plot_obj = (da.plot if plot_func is None else getattr(da.plot, plot_func))(**kwargs)

    # Add coastlines and gridlines
    if isinstance(plot_obj, FacetGrid):
        for ax in plot_obj.axs.flat:
            ax.coastlines()
            ax.gridlines()

        if show_stats:
            warnings.warn(
                "`show_stats` must be False for FacetGrid plots.", UserWarning
            )
    else:
        plot_obj.axes.coastlines()
        if not getattr(plot_obj.axes, "_gridliners", []):
            gl = plot_obj.axes.gridlines(draw_labels=True)
            gl.top_labels = gl.right_labels = False

        # Compute statistics
        if (show_stats is None) or show_stats:
            dataarrays = [diagnostics.spatial_weighted_statistics(da)]
            for stat in "min", "max":
                dataarrays.append(getattr(da, stat)().expand_dims(diagnostic=[stat]))
            da_stats = xr.merge(dataarrays)[da.name]
            n_characters = max(map(len, da_stats["diagnostic"].values.tolist()))

            # Add statistics box
            units = f" [{units}]" if (units := da.attrs.get("units")) else ""
            txt = "\n".join(
                [
                    f"{k:>{n_characters}}: {v.squeeze().values:f}{units}"
                    for k, v in da_stats.groupby("diagnostic")
                ]
            )
            plt.figtext(
                1,
                0.5,
                txt,
                ha="left",
                va="center",
                figure=plot_obj.figure,
                fontfamily="monospace",
            )

    return plot_obj


def _infer_legend_dict(da: xr.DataArray) -> dict[str, tuple[COLOR_T, FLAGS_T]]:
    flags = list(map(int, da.attrs["flag_values"]))
    colors = da.attrs["flag_colors"].split()
    meanings = da.attrs["flag_meanings"].split()

    assert (
        len(flags) == len(meanings) and 0 <= len(flags) - len(colors) <= 1
    ), "flags/meanings/colors mismatch"
    if len(flags) - len(colors) == 1:
        colors.insert(flags.index(0), "#000000")

    legend_dict: dict[str, tuple[COLOR_T, FLAGS_T]] = {}
    for m, c, f in zip(meanings, colors, flags, strict=True):
        legend_dict[m.replace("_", " ").title()] = (c, f)
    return legend_dict


def lccs_map(
    da_lccs: xr.DataArray,
    legend_dict: dict[str, tuple[COLOR_T, FLAGS_T]] | None = None,
    **kwargs: Any,
) -> AxesImage | FacetGrid[Any]:
    """
    Plot LCCS map.

    Parameters
    ----------
    da_lccs: xr.DataArray
        DataArray with LCCS classes
    legend_dict: dict, optional
        Dictionary mapping {meaning: (color, flags)}
    **kwargs:
        Keyword arguments for `da.plot.imshow`

    Returns
    -------
    AxesImage or FacetGrid
    """
    if legend_dict is None:
        legend_dict = _infer_legend_dict(da_lccs)

    # Build vars for plotting
    handles = []
    labels = list(legend_dict)
    color_dict = {}
    for color, flags in legend_dict.values():
        color = matplotlib.colors.to_rgba(color)
        handles.append(matplotlib.patches.Rectangle((0, 0), 1, 1, color=color))
        for flag in [flags] if isinstance(flags, int) else flags:
            assert flag not in color_dict, f"flag {flag} is repeated"
            color_dict[flag] = color

    # Convert to rgb
    rgb = xr.concat(
        [
            xr.DataArray(v, coords=da_lccs.coords, dims=da_lccs.dims)
            for v in np.vectorize(color_dict.get)(da_lccs)
        ],
        "rgb",
    )
    plot_obj = rgb.plot.imshow(rgb="rgb", **kwargs)

    # Add legend
    fig = getattr(plot_obj, "fig" if isinstance(plot_obj, FacetGrid) else "figure")
    fig.legend(handles, labels, bbox_to_anchor=(1, 1), loc="upper left")

    # Add grid
    if isinstance(plot_obj, FacetGrid):
        for ax in plot_obj.axs.flat:
            ax.grid()
    else:
        plot_obj.axes.grid()
    return plot_obj


def lccs_bar(
    da: xr.DataArray,
    da_lccs: xr.DataArray,
    labels_dict: dict[str, tuple[COLOR_T, FLAGS_T]] | None = None,
    reduction: str = "mean",
    groupby_bins_dims: dict[str, Any] = {},
    exclude_no_data: bool = True,
    **kwargs: Any,
) -> Axes:
    """
    Plot LCCS map.

    Parameters
    ----------
    da: xr.DataArray
        DataArray with data to plot
    da_lccs: xr.DataArray
        DataArray with LCCS classes
    labels_dict: dict, optional
        Dictionary mapping {meaning: (color, flags)}
    reduction: str
        Reduction to apply
    groupby_bins_dims: dict
        Dictionary mapping dimension name to bins for groupby_bin_dims
    exclude_no_data: bool
        Whether to exclude the "No Data" flag or not
    **kwargs:
        Keyword arguments for `pd.plot.bar`

    Returns
    -------
    Axes
    """
    if labels_dict is None:
        labels_dict = _infer_legend_dict(da_lccs)

    if exclude_no_data:
        da_lccs = da_lccs.where(da_lccs != 0)

    if groupby_bins_dims:
        assert (
            len(groupby_bins_dims) == 1
        ), "groupby_bins_dims must have a dimension only"
        groupby_dim, *_ = groupby_bins_dims.keys()
        groupby_bins, *_ = groupby_bins_dims.values()

    colors = []
    pd_dict = {}
    for meaning, (color, flags) in labels_dict.items():
        if exclude_no_data and flags in (0, [0], (0,), {0}):
            continue

        masked = da.where(da_lccs.isin([flags] if isinstance(flags, int) else flags))
        if groupby_bins_dims:
            grouped = masked.groupby_bins(groupby_dim, groupby_bins)
            da_reduced = getattr(grouped, reduction)()
            da_reduced = getattr(da_reduced, reduction)(
                set(da.dims) & set(da_reduced.dims)
            )
        else:
            da_reduced = float(getattr(masked, reduction)())

        pd_dict[meaning] = da_reduced
        colors.append(matplotlib.colors.to_hex(color))

    df_or_ser: pd.DataFrame | pd.Series[Any] = (
        pd.DataFrame(pd_dict, index=da_reduced[da_reduced.dims[0]])
        if groupby_bins_dims
        else pd.Series(pd_dict)
    )

    # Set default kwargs
    if groupby_bins_dims:
        kwargs.setdefault("xlabel", xr.plot.utils.label_from_attrs(da[groupby_dim]))
    else:
        kwargs.setdefault("xlabel", xr.plot.utils.label_from_attrs(da_lccs))
    kwargs.setdefault(
        "ylabel",
        " ".join([reduction.title(), "of", xr.plot.utils.label_from_attrs(da)]),
    )

    ax = df_or_ser.plot.bar(color=colors, **kwargs)
    if groupby_bins_dims:
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    return ax


def seasonal_boxplot(
    da: xr.DataArray, time_dim: Hashable | None = None, **kwargs: Any
) -> "pd.Series[Axes]":
    """
    Plot a seasonal boxplot.

    Parameters
    ----------
    da: xr.DataArray
        DataArray to plot
    time: str, optional
        Name of time dimension
    **kwargs: Any
        Keyword arguments for pd.DataFrameGroupBy.boxplot

    Returns
    -------
    Series[Axes]
    """
    kwargs.setdefault("ylabel", xr.plot.utils.label_from_attrs(da))
    kwargs.setdefault("layout", (1, 4))
    if time_dim is None:
        time_dim = utils.get_coord_name(da, "time")

    da = da.stack(stacked_dim=da.dims)
    df = da.to_dataframe()
    axes: pd.Series[Axes] = df.groupby(by=da[time_dim].dt.season.values).boxplot(
        **kwargs
    )
    for ax in axes:
        ax.xaxis.set_ticklabels([])

    return axes
