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


def test_regrid() -> None:
    ds_era5 = download.download_and_transform(*ERA5_REQUEST)
    ds_cmip6 = download.download_and_transform(
        *CMIP6_REQUEST,
        transform_func=diagnostics.regrid,
        transform_func_kwargs={
            "grid_out": ds_era5,
            "method": "bilinear",
            "periodic": True,
        }
    )
    assert dict(ds_cmip6.sizes) == {
        "time": 1,
        "latitude": 721,
        "longitude": 1440,
        "bnds": 2,
    }
