from c3s_eqc_automatic_quality_control import diagnostics, download

CMIP6_REQUEST = (
    "projections-cmip6",
    {
        "format": "zip",
        "temporal_resolution": "monthly",
        "experiment": "historical",
        "variable": "near_surface_air_temperature",
        "model": "cmcc_cm2_sr5",
        "year": "2000",
        "month": "01",
    },
)

ERA5_REQUEST = (
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "format": "netcdf",
        "time": "00:00",
        "variable": "2m_temperature",
        "year": "2000",
        "month": "01",
    },
)


def test_weighted_diagnostics() -> None:
    ds_era5 = download.download_and_transform(*ERA5_REQUEST)

    assert dict(diagnostics.spatial_weighted_statistics(ds_era5).sizes) == {
        "time": 1,
        "diagnostic": 3,
    }
    assert dict(diagnostics.spatial_weighted_errors(ds_era5, ds_era5).sizes) == {
        "time": 1,
        "diagnostic": 3,
    }

    assert dict(diagnostics.seasonal_weighted_mean(ds_era5).sizes) == {
        "longitude": 1440,
        "latitude": 721,
        "season": 1,
    }
    assert dict(diagnostics.annual_weighted_mean(ds_era5).sizes) == {
        "longitude": 1440,
        "latitude": 721,
    }


def test_regrid() -> None:
    ds_era5 = download.download_and_transform(*ERA5_REQUEST)
    ds_cmip6 = download.download_and_transform(*CMIP6_REQUEST)

    ds_reg = diagnostics.regrid(
        ds_cmip6, grid_out=ds_era5, method="bilinear", periodic=True
    )
    assert dict(ds_reg.sizes) == {
        "time": 1,
        "latitude": 721,
        "longitude": 1440,
        "bnds": 2,
    }
