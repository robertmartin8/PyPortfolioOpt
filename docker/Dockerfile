FROM python:3.11-slim-buster

WORKDIR pypfopt
COPY requirements.txt .

RUN pip install --upgrade pip \
    pip install yfinance && \
    pip install poetry \
    pip install ipython \
    pip install jupyter \
    pip install pytest \
    pip install -r requirements.txt

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
