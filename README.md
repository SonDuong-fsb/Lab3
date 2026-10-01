# Lab 3: Testing & CI/CD for ML Systems

[![CI Pipeline](https://github.com/SonDuong-fsb/Lab3/actions/workflows/ci.yml/badge.svg)](https://github.com/SonDuong-fsb/Lab3/actions/workflows/ci.yml)

Testing strategy and CI/CD pipelines for the movie rating prediction system (FastAPI + SVD collaborative filtering on MovieLens 100K).

**Course:** DDM501 - AI in Production: From Models to Systems
**Weight:** 15% of total grade
**Prerequisites:** Lab 1 and Lab 2 completed

## Status

| Area | Result |
|------|--------|
| Tests | 175 passing (69 unit, 46 integration, 31 data, 29 model) |
| Coverage | 100% of `app/` (gate: 80%) |
| flake8 / black / isort | clean |
| mypy | clean |
| Pre-commit | black, isort, flake8, mypy, unit tests, file hygiene |
| Workflows | `ci.yml`, `cd.yml`, `model-validation.yml` |

Details and rationale: [docs/TESTING_STRATEGY.md](docs/TESTING_STRATEGY.md).

## Project Structure

```
Lab3/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── model.py                # ML model wrapper
│   ├── schemas.py              # Pydantic schemas
│   └── config.py               # Configuration
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # test_model.py, test_schemas.py
│   ├── integration/            # test_api.py
│   ├── data/                   # test_data_quality.py
│   └── model/                  # test_model_behavior.py
├── .github/workflows/
│   ├── ci.yml                  # lint, type-check, test, docker build
│   ├── cd.yml                  # tag-triggered image publish and release
│   └── model-validation.yml    # retrain and validate model/data
├── docs/
│   └── TESTING_STRATEGY.md
├── scripts/train_model.py      # Model training script
├── models/                     # Saved models (git-ignored)
├── .pre-commit-config.yaml
├── .flake8
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## Quick Start

### 1. Setup

Python 3.10 is the supported version (it is what CI uses, and `scikit-surprise==1.1.3` compiles against the pinned `numpy`).
Building `scikit-surprise` needs a C compiler (`build-essential` on Debian/Ubuntu, MSVC Build Tools on Windows).

```bash
git clone https://github.com/SonDuong-fsb/Lab3.git
cd Lab3

python -m venv venv
source venv/bin/activate

bash scripts/install_requirements.sh
pip install -r requirements-dev.txt
```

The script installs `numpy` first and then builds `scikit-surprise` with `--no-build-isolation`. A plain `pip install -r requirements.txt` fails on current pip, because the `scikit-surprise` 1.1.3 build script calls `pip` from inside pip's isolated build environment.

### 2. Train the model

Downloads MovieLens 100K to `~/.surprise_data` on first run and writes `models/svd_model.pkl`.

```bash
python scripts/train_model.py
```

### 3. Run the tests

```bash
pytest tests/ --cov=app --cov-report=term-missing --cov-report=html

pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/data/ -v
pytest tests/model/ -v
```

Tests that need the trained model or the MovieLens file are skipped with a message if those are missing, so train first.
Open `htmlcov/index.html` for the coverage report.

### 4. Code quality

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files

black app/ tests/ scripts/
isort app/ tests/ scripts/
flake8 app/ tests/ scripts/
mypy app/
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API information |
| `/health` | GET | Liveness and model-loaded flag |
| `/predict` | POST | Rating for one `user_id` / `movie_id` |
| `/predict/batch` | POST | Ratings for 1-100 pairs |
| `/model/info` | GET | Model version and load state |

### 6. Run with Docker

The image bundles `models/svd_model.pkl`, so train the model first.

```bash
docker build -t movie-rating-api .
docker run -p 8000:8000 movie-rating-api
curl http://localhost:8000/health
```

## CI/CD

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | push to `main`/`develop`, PR to `main` | flake8, black, isort and mypy in parallel; then trains the model and runs pytest with an 80% coverage gate; then builds the Docker image and smoke-tests `/health` and `/predict` |
| `cd.yml` | tag `v*` | trains, runs the test gate, pushes `:latest` and `:vX.Y.Z` to Docker Hub, creates a GitHub release, then pulls the published image and smoke-tests it |
| `model-validation.yml` | changes to `models/`, `scripts/`, `app/model.py`, `tests/model/`, or manual | retrains and runs the model and data suites |

Repository secrets required by `cd.yml`: `DOCKER_USERNAME`, `DOCKER_PASSWORD`. `CODECOV_TOKEN` is optional.

To cut a release: `git tag v1.0.0 && git push origin v1.0.0`. To roll back, redeploy the previous `:vX.Y.Z` image.

## Testing Summary

| Layer | Focus |
|-------|-------|
| Unit | Schema validation, model loading, rounding/clipping, error handling |
| Integration | Every endpoint, validation errors, 404/405/422/500/503, startup |
| Data | Ranges, nulls, types, uniqueness and distribution of the MovieLens data |
| Model | Invariance, directional, minimum functionality, MAE/RMSE thresholds, robustness |

## Submission Checklist

- [x] Unit, integration, data and model tests implemented
- [x] CI workflow, CD workflow and model-validation workflow
- [x] Pre-commit hooks configured
- [x] Coverage above 80%
- [x] Testing strategy document
- [x] Badge URLs point to SonDuong-fsb/Lab3
- [ ] Add screenshots of passing workflow runs

## License

MIT License - For educational purposes only.
