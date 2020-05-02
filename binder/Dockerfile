FROM jupyter/base-notebook:lab-1.2.5 as jupyter

# File Author / Maintainer
# MAINTAINER Thomas Schmelzer "thomas.schmelzer@gmail.com"

# copy the config file
COPY ./binder/jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py

# copy the package over and install it
COPY --chown=jovyan:users . /tmp/pyportfolioopt

RUN conda install -y -c conda-forge --file /tmp/pyportfolioopt/requirements.txt && \
    conda clean -y --all && \
    pip install --no-cache-dir /tmp/pyportfolioopt && \
    rm -rf /tmp/pyportfolioopt && \
    pip install yfinance && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install beakerx-jupyterlab && \
    jupyter lab build

# hardcoded parameters!? see https://github.com/moby/moby/issues/35018
COPY --chown=jovyan:users ./cookbook $HOME/work
