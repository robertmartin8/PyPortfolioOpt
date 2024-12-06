FROM python:3.12

WORKDIR /pypfopt
COPY --link requirements.txt .

# prepare the system to install scs
RUN apt-get update && apt-get install -y \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip && \
    pip install yfinance \
    poetry \
    ipython \
    jupyter \
    ecos \
    pytest && \
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
