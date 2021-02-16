FROM python:3.8-slim-buster

WORKDIR pypfopt
COPY ../pyproject.toml ../poetry.lock ./

RUN buildDeps='gcc g++' && \
    apt-get update && apt-get install -y $buildDeps --no-install-recommends && \
    pip install --upgrade pip==21.0.1 && \
    pip install "poetry==1.1.4" && \
    poetry install -E optionals --no-root && \
    apt-get purge -y --auto-remove $buildDeps

COPY ../. .

# Usage examples:
#
# Build
# docker build . -t pypfopt
#
# Run
# iPython interpreter:
# docker run -it pypfopt poetry run ipython
# Jupyter notebook server:
# docker run -it -p 8888:8888 pypfopt poetry run jupyter notebook --allow-root --no-browser --ip 0.0.0.0
# Pytest
# docker run -t pypfopt poetry run pytest
# Bash
# docker run -it pypfopt bash
