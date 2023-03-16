import json
import os
import pathlib

from c3s_eqc_automatic_quality_control import dashboard, runner

TEST_STR = "Hello World!"
SAMPLE_NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 0,
            "outputs": [],
            "source": [f"print('{TEST_STR}')"],
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.8",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def test_run(tmp_path: pathlib.Path) -> None:
    notebook = "notebook.ipynb"
    temp_config = tmp_path / notebook
    temp_config.write_text(json.dumps(SAMPLE_NOTEBOOK))
    qar_id = "0"
    run_n = "0"

    old_environ = os.environ
    try:
        os.environ[dashboard.EQC_AQC_ENV_VARNAME] = str(tmp_path)
        runner.run(
            notebook_path=str(temp_config),
            target_dir=str(tmp_path),
            qar_id=qar_id,
            run_n=run_n,
        )
    finally:
        os.environ.clear()
        os.environ.update(old_environ)

    run_folder = tmp_path / f"qar_{qar_id}" / f"run_{run_n}"
    assert set(run_folder.glob("*")) == {run_folder / notebook}
    with open(str(run_folder / notebook)) as f:
        nb = json.load(f)
        print(nb["cells"][0]["outputs"][0]["text"])
        assert TEST_STR in nb["cells"][0]["outputs"][0]["text"][0]
