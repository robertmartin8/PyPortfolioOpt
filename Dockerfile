FROM python:3.7.7-slim-stretch as builder

# File Author / Maintainer
# MAINTAINER

# this will be user root regardless whether home/beakerx is not
COPY . /tmp/pypfopt

RUN buildDeps='gcc g++' && \
    apt-get update && apt-get install -y $buildDeps --no-install-recommends && \
    pip install --no-cache-dir -r /tmp/pypfopt/requirements.txt && \
    # One could install the pypfopt library directly in the image. We don't and share via docker-compose instead.
    # pip install --no-cache-dir /tmp/pyhrp && \
    rm -r /tmp/pypfopt && \
    apt-get purge -y --auto-remove $buildDeps


# ----------------------------------------------------------------------------------------------------------------------
FROM builder as test

# COPY tools needed for testing into the image
RUN pip install --no-cache-dir  pytest pytest-cov pytest-html

# COPY the tests over
COPY tests /pypfopt/tests

WORKDIR /pypfopt

CMD py.test --cov=pypfopt  --cov-report html:artifacts/html-coverage --cov-report term --html=artifacts/html-report/report.html tests
