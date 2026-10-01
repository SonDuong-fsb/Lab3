import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


def build_response(rating, user_id="196", movie_id="242"):
    return PredictionResponse(
        user_id=user_id,
        movie_id=movie_id,
        predicted_rating=rating,
        model_version="1.0.0",
    )


class TestPredictionRequest:
    def test_valid_request(self):
        request = PredictionRequest(user_id="196", movie_id="242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_valid_request_with_numeric_strings(self):
        request = PredictionRequest(user_id="123", movie_id="456")
        assert request.user_id == "123"
        assert request.movie_id == "456"

    def test_missing_user_id_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(movie_id="242")

    def test_missing_movie_id_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196")

    def test_missing_both_fields_raises_error(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest()
        assert len(exc_info.value.errors()) == 2

    def test_empty_user_id_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="", movie_id="242")

    def test_empty_movie_id_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="196", movie_id="")

    def test_whitespace_only_user_id_raises_error(self):
        with pytest.raises(ValidationError, match="empty or whitespace"):
            PredictionRequest(user_id="   ", movie_id="242")

    def test_whitespace_only_movie_id_raises_error(self):
        with pytest.raises(ValidationError, match="empty or whitespace"):
            PredictionRequest(user_id="196", movie_id="\t")

    def test_none_values_raise_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=None, movie_id=None)

    def test_surrounding_whitespace_is_stripped(self):
        request = PredictionRequest(user_id="  196 ", movie_id=" 242")
        assert request.user_id == "196"
        assert request.movie_id == "242"

    def test_id_at_maximum_length_is_accepted(self):
        request = PredictionRequest(user_id="1" * 50, movie_id="242")
        assert len(request.user_id) == 50

    def test_id_above_maximum_length_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id="1" * 51, movie_id="242")

    def test_integer_user_id_is_rejected(self):
        with pytest.raises(ValidationError):
            PredictionRequest(user_id=196, movie_id="242")

    def test_integer_ids_are_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(user_id=196, movie_id=242)
        assert len(exc_info.value.errors()) == 2


class TestPredictionResponse:
    def test_valid_response(self):
        response = build_response(3.5)
        assert response.predicted_rating == 3.5
        assert response.model_version == "1.0.0"

    def test_rating_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            build_response(0.5)

    def test_rating_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            build_response(5.5)

    @pytest.mark.parametrize("rating", [1.0, 5.0])
    def test_rating_at_boundaries(self, rating):
        assert build_response(rating).predicted_rating == rating

    def test_missing_model_version_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionResponse(user_id="196", movie_id="242", predicted_rating=3.0)


class TestHealthResponse:
    def test_valid_health_response(self):
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True

    @pytest.mark.parametrize("status", ["healthy", "unhealthy", "degraded"])
    def test_health_response_status_types(self, status):
        assert HealthResponse(status=status, model_loaded=False).status == status

    def test_model_loaded_must_be_boolean_like(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy", model_loaded="maybe")

    def test_missing_fields_raise_error(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="healthy")


class TestPredictionItem:
    def test_valid_item(self):
        item = PredictionItem(user_id="196", movie_id="242")
        assert (item.user_id, item.movie_id) == ("196", "242")

    def test_empty_ids_raise_error(self):
        with pytest.raises(ValidationError):
            PredictionItem(user_id="", movie_id="")


class TestBatchPredictionRequest:
    def test_valid_batch_request(self):
        request = BatchPredictionRequest(
            predictions=[
                {"user_id": "196", "movie_id": "242"},
                {"user_id": "186", "movie_id": "302"},
            ]
        )
        assert len(request.predictions) == 2
        assert all(isinstance(item, PredictionItem) for item in request.predictions)

    def test_empty_predictions_list_raises_error(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[])

    def test_too_many_predictions_raises_error(self):
        items = [{"user_id": "1", "movie_id": str(index + 1)} for index in range(101)]
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=items)

    def test_maximum_batch_size_is_accepted(self):
        items = [{"user_id": "1", "movie_id": str(index + 1)} for index in range(100)]
        assert len(BatchPredictionRequest(predictions=items).predictions) == 100

    def test_invalid_item_raises_error(self):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(predictions=[{"user_id": "196"}])


class TestBatchPredictionResponse:
    def test_valid_batch_response(self):
        response = BatchPredictionResponse(
            predictions=[build_response(3.5), build_response(4.0, movie_id="302")],
            total_count=2,
        )
        assert response.total_count == len(response.predictions)

    def test_nested_rating_is_validated(self):
        invalid = {
            "user_id": "196",
            "movie_id": "242",
            "predicted_rating": 9.0,
            "model_version": "1.0.0",
        }
        with pytest.raises(ValidationError):
            BatchPredictionResponse(predictions=[invalid], total_count=1)


class TestErrorResponse:
    def test_default_error_code(self):
        assert ErrorResponse(detail="boom").error_code == "UNKNOWN_ERROR"

    def test_custom_error_code(self):
        response = ErrorResponse(detail="boom", error_code="MODEL_UNAVAILABLE")
        assert response.error_code == "MODEL_UNAVAILABLE"
