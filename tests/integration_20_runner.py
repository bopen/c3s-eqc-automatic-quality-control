import os
import pathlib

from c3s_eqc_automatic_quality_control import dashboard, runner

EQC_CONFIG = """
qar_id: 0
run_n: 0
collection_id: reanalysis-era5-single-levels
product_type:  reanalysis
format: grib
time: [00]
variables:
  - 2m_temperature
start: 2020-01
stop: 2020-02
diagnostics:
  - spatial_weighted_mean
"""


def test_run(tmp_path: pathlib.Path) -> None:
    temp_config = tmp_path / "temp_eqc_config.yml"
    temp_config.write_text(EQC_CONFIG)

    old_environ = os.environ
    try:
        os.environ[dashboard.EQC_AQC_ENV_VARNAME] = str(tmp_path)
        runner.run(str(temp_config), str(tmp_path))
    finally:
        os.environ.clear()
        os.environ.update(old_environ)

    run_folder = tmp_path / "qar_0" / "run_0"
    assert set(run_folder.glob("*")) == {
        run_folder / "2m_temperature_spatial_weighted_mean_image.png",
        run_folder / "2m_temperature_spatial_weighted_mean_metadata.json",
    }
