FROM continuumio/miniconda3

WORKDIR /src/c3s-eqc-automatic-quality-control

COPY environment.yml /src/c3s-eqc-automatic-quality-control/

RUN conda install -c conda-forge gcc python=3.11 \
    && conda env update -n base -f environment.yml

COPY . /src/c3s-eqc-automatic-quality-control

RUN pip install --no-deps -e .
