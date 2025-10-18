# Makefile - dev helper for environment, examples and tests
VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
ACTIVATE=. $(VENV)/bin/activate

.PHONY: venv install deps pio-init pio-build pio-upload flash-esp32 \
        run-f9p run-pixhawk run-emotiv test clean

venv:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel

install: venv
	$(VENV)/bin/pip install -e .
	$(VENV)/bin/pip install -r dev-requirements.txt || true

deps: install
	$(VENV)/bin/pip install pyserial pymavlink pyubx2 rplidar brainflow platformio
