#!make
PROJECT_VERSION := $(shell python setup.py --version)

SHELL := /bin/bash
PACKAGE := pypfopt

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



build:
	docker-compose build pypfopt

test:
	mkdir -p artifacts
	docker-compose -f docker-compose.test.yml run sut

tag: test
	git tag -a ${PROJECT_VERSION} -m "new tag"
	git push --tags

#pypi: tag
#	python setup.py sdist
#	twine check dist/*
#	twine upload dist/*