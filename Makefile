.PHONY: test pipeline

test:
	python -m pytest -q

pipeline:
	python pipeline.py ${ARGS}

test_helpers:
	python -m pytest -q tests/test_helpers.py

test_db:
	python -m pytest -q tests/test_db.py
