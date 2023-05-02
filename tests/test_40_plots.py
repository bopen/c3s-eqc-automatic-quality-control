import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.geocollection import GeoQuadMesh
from xarray.plot.facetgrid import FacetGrid

from c3s_eqc_automatic_quality_control import plot


def test_projected_map() -> None:
    ds = xr.tutorial.open_dataset("air_temperature")

    actual = plot.projected_map(ds["air"].isel(time=0))
    assert isinstance(actual, GeoQuadMesh)

    actual = plot.projected_map(ds["air"].isel(time=slice(2)), col="time")
    assert isinstance(actual, FacetGrid)
    for ax in actual.axs.flat:
        assert isinstance(ax, GeoAxes)
