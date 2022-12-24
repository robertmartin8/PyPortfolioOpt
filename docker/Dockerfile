FROM python:3.8-buster

WORKDIR pypfopt
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip \
    pip install poetry yfinance && \
    poetry install -E optionals --no-root 

COPY . .

RUN cd cookbook

# Usage examples:
#
# Build
# from root of repo:
# docker build -f docker/Dockerfile . -t pypfopt
#
# Run
# iPython interpreter:
# docker run -it pypfopt poetry run ipython
# Jupyter notebook server:
# docker run -it -p 8888:8888 pypfopt poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0
# click on http://127.0.0.1:8888/?token=xxx
# Pytest
# docker run -t pypfopt poetry run pytest
# Bash
# docker run -it pypfopt bash
