#!make
PROJECT_VERSION := $(shell python setup.py --version)

SHELL := /bin/bash
PACKAGE := pypfopt

# needed to get the ${PORT} environment variable
include .env
export

.PHONY: help build test tag pypi


.DEFAULT: help

help:
	@echo "make build"
	@echo "       Build the docker image."
	@echo "make test"
	@echo "       Build the docker image for testing and run them."
	@echo "make doc"
	@echo "       Construct the documentation."
	@echo "make tag"
	@echo "       Make a tag on Github."

jupyter:
	echo "http://localhost:${PORT}"
	docker-compose up jupyter

build:
	docker-compose build pypfopt

test:
	mkdir -p artifacts
	docker-compose -f docker-compose.test.yml run sut

tag: test
	git tag -a ${PROJECT_VERSION} -m "new tag"
	git push --tags

# push to pypi
#pypi: tag
#	python setup.py sdist
#	twine check dist/*
#	twine upload dist/*

# push to dockerhub
# you will need a free account on dockerhub and do docker login...
#hub: tag
#	docker build --file binder/Dockerfile --tag ${IMAGE}:latest --no-cache .
#	docker push ${IMAGE}:latest
#	docker tag ${IMAGE}:latest ${IMAGE}:${PROJECT_VERSION}
#	docker push ${IMAGE}:${PROJECT_VERSION}
#	docker rmi -f ${IMAGE}:${PROJECT_VERSION}