import os
import pathlib
import tempfile

from c3s_eqc_automatic_quality_control import dashboard


def test_get_logger() -> None:
    logger_name = "test-logger"
    logger = dashboard.get_logger(logger_name)
    assert logger.name == logger_name


def test_set_log_file() -> None:
    logger_name = "test-logger"
    logger_text = "TEST EQC"
    logger = dashboard.get_logger(logger_name)
    with tempfile.NamedTemporaryFile(delete=False) as t:
        name = t.name
        logger = dashboard.set_logfile(logger, pathlib.Path(name))
        logger.info(logger_text)
    with open(name) as f:
        assert logger_text in f.readline()
    os.remove(name)
    assert not os.path.exists(name)


def test_ensure_log_dir() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        original_eqc_etc = os.environ.get(dashboard.EQC_AQC_ENV_VARNAME)
        os.environ[dashboard.EQC_AQC_ENV_VARNAME] = tempdir
        log_dir = dashboard.ensure_log_dir()
        assert log_dir.is_dir()
        if original_eqc_etc is None:
            os.environ.pop(dashboard.EQC_AQC_ENV_VARNAME)
        else:
            os.environ[dashboard.EQC_AQC_ENV_VARNAME] = original_eqc_etc


def test_get_eqc_run_logger() -> None:
    logger_name = "logger-name"
    logger = dashboard.get_eqc_run_logger(logger_name)
    assert f"_{logger_name}.log" in getattr(logger.handlers[0], "baseFilename")
