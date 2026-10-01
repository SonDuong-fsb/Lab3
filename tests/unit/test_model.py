import pickle
from types import SimpleNamespace

import pytest

from app import model as model_module
from app.config import MAX_RATING, MIN_RATING
from app.model import MovieRatingModel, get_model, reset_model


class StubEstimator:
    def __init__(self, estimate):
        self.estimate = estimate

    def predict(self, user_id, movie_id):
        return SimpleNamespace(est=self.estimate)


@pytest.fixture
def stubbed_model(monkeypatch):
    monkeypatch.setattr(MovieRatingModel, "_load_model", lambda self: None)
    return MovieRatingModel(model_path="unused.pkl")


class TestMovieRatingModel:
    def test_model_loads_successfully(self, trained_model):
        assert trained_model is not None
        assert trained_model.is_loaded()

    def test_model_instance_has_model_attribute(self, trained_model):
        assert hasattr(trained_model, "model")
        assert trained_model.model is not None

    def test_model_remembers_its_path(self, trained_model):
        assert trained_model.model_path.endswith(".pkl")

    def test_predict_returns_float(self, trained_model):
        result = trained_model.predict("196", "242")
        assert isinstance(result, float)

    def test_predict_returns_value_in_valid_range(self, trained_model):
        result = trained_model.predict("196", "242")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_rounds_to_two_decimals(self, trained_model):
        result = trained_model.predict("196", "242")
        assert result == round(result, 2)

    def test_predict_multiple_pairs_all_in_range(self, trained_model, known_user_movie_pairs):
        for pair in known_user_movie_pairs:
            result = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert MIN_RATING <= result <= MAX_RATING

    def test_predict_batch_returns_list(self, trained_model):
        pairs = [("196", "242"), ("186", "302")]
        results = trained_model.predict_batch(pairs)
        assert isinstance(results, list)

    def test_predict_batch_returns_correct_length(self, trained_model):
        pairs = [("196", "242"), ("186", "302"), ("22", "377")]
        results = trained_model.predict_batch(pairs)
        assert len(results) == len(pairs)

    def test_predict_batch_all_values_in_range(self, trained_model):
        pairs = [("196", "242"), ("186", "302"), ("22", "377"), ("99999", "99999")]
        results = trained_model.predict_batch(pairs)
        assert all(MIN_RATING <= rating <= MAX_RATING for rating in results)

    def test_predict_batch_empty_input_returns_empty_list(self, trained_model):
        assert trained_model.predict_batch([]) == []

    def test_predict_batch_matches_single_predictions(self, trained_model):
        pairs = [("196", "242"), ("186", "302")]
        expected = [trained_model.predict(user, movie) for user, movie in pairs]
        assert trained_model.predict_batch(pairs) == expected

    def test_is_loaded_returns_bool(self, trained_model):
        assert isinstance(trained_model.is_loaded(), bool)

    def test_is_loaded_returns_true_for_loaded_model(self, trained_model):
        assert trained_model.is_loaded() is True

    def test_predict_with_none_user_id(self, trained_model):
        result = trained_model.predict(None, "242")
        assert MIN_RATING <= result <= MAX_RATING

    def test_predict_with_empty_string(self, trained_model):
        result = trained_model.predict("", "")
        assert MIN_RATING <= result <= MAX_RATING

    def test_unknown_user_gets_same_prediction_as_other_unknown_user(self, trained_model):
        assert trained_model.predict("99999", "242") == trained_model.predict("88888", "242")


class TestPredictionClipping:
    @pytest.mark.parametrize(
        ("estimate", "expected"),
        [
            (7.3, MAX_RATING),
            (5.0, 5.0),
            (3.456, 3.46),
            (1.0, 1.0),
            (-2.4, MIN_RATING),
        ],
    )
    def test_prediction_is_rounded_and_clipped(self, stubbed_model, estimate, expected):
        stubbed_model.model = StubEstimator(estimate)
        assert stubbed_model.predict("1", "1") == expected


class TestUnloadedModel:
    def test_is_loaded_is_false_without_estimator(self, stubbed_model):
        assert stubbed_model.is_loaded() is False

    def test_predict_raises_without_estimator(self, stubbed_model):
        with pytest.raises(RuntimeError, match="not loaded"):
            stubbed_model.predict("196", "242")

    def test_predict_batch_raises_without_estimator(self, stubbed_model):
        with pytest.raises(RuntimeError, match="not loaded"):
            stubbed_model.predict_batch([("196", "242")])


class TestModelFileHandling:
    def test_model_raises_error_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            MovieRatingModel(model_path="/nonexistent/path/model.pkl")

    def test_model_raises_error_for_corrupt_file(self, tmp_path):
        corrupt_file = tmp_path / "corrupt.pkl"
        corrupt_file.write_bytes(b"not a pickle")
        with pytest.raises(pickle.UnpicklingError):
            MovieRatingModel(model_path=str(corrupt_file))

    def test_model_loads_from_custom_path(self, tmp_path):
        custom_file = tmp_path / "custom.pkl"
        custom_file.write_bytes(pickle.dumps({"kind": "placeholder"}))
        loaded = MovieRatingModel(model_path=str(custom_file))
        assert loaded.is_loaded()
        assert loaded.model == {"kind": "placeholder"}


class TestModelSingleton:
    @pytest.fixture(autouse=True)
    def clean_singleton(self):
        reset_model()
        yield
        reset_model()

    def test_get_model_returns_same_instance(self, trained_model):
        assert get_model() is get_model()

    def test_reset_model_discards_instance(self, trained_model):
        first = get_model()
        reset_model()
        assert get_model() is not first

    def test_reset_model_clears_module_state(self, trained_model):
        get_model()
        reset_model()
        assert model_module._model_instance is None
