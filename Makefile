SHELL := /bin/bash

.PHONY: all
all: format lint

.PHONY: format
format:
	ruff format src scripts

.PHONY: lint
lint:
	ruff check src scripts

