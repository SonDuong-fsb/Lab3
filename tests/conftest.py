import os
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_PATH
from app.main import app
from app.model import MovieRatingModel

DATA_DIR = Path(os.getenv("SURPRISE_DATA_FOLDER", str(Path.home() / ".surprise_data")))
RATINGS_FILE = DATA_DIR / "ml-100k" / "ml-100k" / "u.data"
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


@pytest.fixture(scope="session")
def test_client() -> Iterator[TestClient]:
    if not Path(MODEL_PATH).exists():
        pytest.skip("Model file not found. Run scripts/train_model.py first.")
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def trained_model() -> MovieRatingModel:
    try:
        return MovieRatingModel()
    except FileNotFoundError:
        pytest.skip("Model file not found. Run scripts/train_model.py first.")


@pytest.fixture(scope="session")
def movielens_ratings() -> pd.DataFrame:
    if not RATINGS_FILE.exists():
        pytest.skip("MovieLens 100K data not found. Run scripts/train_model.py first.")
    return pd.read_csv(
        RATINGS_FILE,
        sep="\t",
        names=RATING_COLUMNS,
        dtype={"user_id": str, "movie_id": str},
    )


@pytest.fixture
def sample_prediction_request():
    return {"user_id": "196", "movie_id": "242"}


@pytest.fixture
def sample_batch_request():
    return {
        "predictions": [
            {"user_id": "196", "movie_id": "242"},
            {"user_id": "186", "movie_id": "302"},
            {"user_id": "22", "movie_id": "377"},
        ]
    }


@pytest.fixture
def sample_ratings():
    return [
        {"user_id": "1", "movie_id": "10", "rating": 4.0},
        {"user_id": "1", "movie_id": "20", "rating": 3.5},
        {"user_id": "2", "movie_id": "10", "rating": 5.0},
        {"user_id": "2", "movie_id": "30", "rating": 2.0},
        {"user_id": "3", "movie_id": "10", "rating": 3.0},
        {"user_id": "3", "movie_id": "20", "rating": 4.5},
        {"user_id": "3", "movie_id": "30", "rating": 1.0},
    ]


@pytest.fixture
def invalid_prediction_requests():
    return [
        {},
        {"user_id": "196"},
        {"movie_id": "242"},
        {"user_id": "", "movie_id": "242"},
        {"user_id": "196", "movie_id": ""},
        {"user_id": "   ", "movie_id": "242"},
    ]


@pytest.fixture
def known_user_movie_pairs():
    return [
        {"user_id": "196", "movie_id": "242", "actual_rating": 3.0},
        {"user_id": "186", "movie_id": "302", "actual_rating": 3.0},
        {"user_id": "22", "movie_id": "377", "actual_rating": 1.0},
        {"user_id": "244", "movie_id": "51", "actual_rating": 2.0},
        {"user_id": "166", "movie_id": "346", "actual_rating": 1.0},
    ]


@pytest.fixture
def unknown_users():
    return ["99999", "999999", "0", "-1", "new_user"]


@pytest.fixture
def unknown_movies():
    return ["99999", "999999", "0", "-1", "new_movie"]
