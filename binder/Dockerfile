FROM jupyter/base-notebook:python-3.7.6 as jupyter

# File Author / Maintainer
# MAINTAINER Thomas Schmelzer "thomas.schmelzer@gmail.com"

# copy the config file
COPY ./binder/jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py

# copy the package over and install it
COPY --chown=jovyan:users . /tmp/pyportfolioopt

RUN conda install -y -c conda-forge --file /tmp/pyportfolioopt/requirements.txt && \
    conda clean -y --all && \
    pip install --no-cache-dir /tmp/pyportfolioopt && \
    pip install yfinance && \
    pip install lxml && \
    rm -rf /tmp/pyportfolioopt

# hardcoded parameters!? see https://github.com/moby/moby/issues/35018
COPY --chown=jovyan:users ./cookbook $HOME/work
