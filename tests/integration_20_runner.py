import glob
import os
import tempfile

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


def test_run() -> None:
    current = os.getcwd()
    with tempfile.TemporaryDirectory() as tempdir:
        temp_config = tempdir + "/temp_eqc_config.yml"
        with open(temp_config, "w") as f:
            f.write(EQC_CONFIG)
        original_eqc_etc = os.environ.get(dashboard.EQC_AQC_ENV_VARNAME)
        os.environ[dashboard.EQC_AQC_ENV_VARNAME] = tempdir
        runner.run(temp_config, tempdir)
        if original_eqc_etc is None:
            os.environ.pop(dashboard.EQC_AQC_ENV_VARNAME)
        else:
            os.environ[dashboard.EQC_AQC_ENV_VARNAME] = original_eqc_etc
        run_folder = f"{tempdir}/qar_0/run_0"
        assert os.path.isdir(run_folder)
        os.chdir(run_folder)
        assert "2m_temperature_spatial_weighted_mean.png" in glob.glob("*.png")
        assert "2m_temperature_metadata.json" in glob.glob("*.json")
    os.chdir(current)
