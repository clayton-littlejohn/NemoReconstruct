VENV      := .venv
VENV_BIN  := $(VENV)/bin
PYTHON    := $(VENV_BIN)/python
PIP       := $(VENV_BIN)/pip

.PHONY: backend-install backend-dev frontend-install frontend-dev openapi setup download-datasets list-datasets

# ── First-time setup ──────────────────────────────────────────────
setup: $(VENV)/pyvenv.cfg backend-install frontend-install

$(VENV)/pyvenv.cfg:
	python3 -m venv $(VENV)

backend-install: $(VENV)/pyvenv.cfg
	$(PIP) install -r backend/requirements.txt

backend-dev: $(VENV)/pyvenv.cfg
	cd backend && $(CURDIR)/$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8010 --reload

frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev

openapi: $(VENV)/pyvenv.cfg
	cd backend && $(CURDIR)/$(PYTHON) export_openapi.py

# ── Dataset management ────────────────────────────────────────────
download-datasets:
	./scripts/download_datasets.sh

download-dataset-%:
	./scripts/download_datasets.sh $*

list-datasets:
	./scripts/download_datasets.sh --list
