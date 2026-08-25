VENV := venv
PY   := $(VENV)/bin/python3.12
PIP  := $(VENV)/bin/pip

MODEL ?= prism-ml/Bonsai-1.7B-unpacked
BASE  ?= Qwen/Qwen3-1.7B

.PHONY: all install run info compare flips profile demo clean

all: install

install:
	python3.12 -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

run:
	$(PY) main.py --help

info:
	$(PY) main.py info $(MODEL)

compare:
	$(PY) main.py compare $(MODEL) $(BASE) model.layers.0.mlp.gate_proj.weight

flips:
	$(PY) main.py flips $(MODEL) $(BASE) model.layers.0.mlp.gate_proj.weight

profile:
	$(PY) main.py profile $(MODEL) $(BASE)

demo: info compare flips profile

clean:
	rm -rf $(VENV) __pycache__ bitprobe/__pycache__ *.pyc
