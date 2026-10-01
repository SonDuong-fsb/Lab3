import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import API_TITLE, API_VERSION, MODEL_VERSION

pytestmark = pytest.mark.integration


class BrokenModel:
    def is_loaded(self):
        return True

    def predict(self, user_id, movie_id):
        raise ValueError("inference failed")


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_status_field(self, test_client):
        response = test_client.get("/health")
        assert "status" in response.json()

    def test_health_response_has_model_loaded_field(self, test_client):
        response = test_client.get("/health")
        assert "model_loaded" in response.json()

    def test_health_model_loaded_is_boolean(self, test_client):
        response = test_client.get("/health")
        assert isinstance(response.json()["model_loaded"], bool)

    def test_health_reports_healthy_when_model_is_loaded(self, test_client):
        assert test_client.get("/health").json() == {"status": "healthy", "model_loaded": True}

    def test_health_reports_unhealthy_when_model_is_missing(self, test_client, monkeypatch):
        monkeypatch.setattr(main, "model", None)
        assert test_client.get("/health").json() == {"status": "unhealthy", "model_loaded": False}


class TestRootEndpoint:
    def test_root_returns_200(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self, test_client):
        data = test_client.get("/").json()
        assert data["name"] == API_TITLE
        assert data["version"] == API_VERSION
        assert data["docs"] == "/docs"
        assert data["health"] == "/health"


class TestPredictEndpoint:
    def test_predict_valid_request_returns_200(self, test_client, sample_prediction_request):
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200

    def test_predict_response_has_predicted_rating(self, test_client, sample_prediction_request):
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert "predicted_rating" in data

    def test_predict_response_has_user_id(self, test_client, sample_prediction_request):
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["user_id"] == sample_prediction_request["user_id"]

    def test_predict_response_has_movie_id(self, test_client, sample_prediction_request):
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["movie_id"] == sample_prediction_request["movie_id"]

    def test_predict_response_has_model_version(self, test_client, sample_prediction_request):
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert data["model_version"] == MODEL_VERSION

    def test_predict_response_rating_in_valid_range(self, test_client, sample_prediction_request):
        data = test_client.post("/predict", json=sample_prediction_request).json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_matches_model_output(self, test_client, trained_model):
        payload = {"user_id": "186", "movie_id": "302"}
        data = test_client.post("/predict", json=payload).json()
        assert data["predicted_rating"] == trained_model.predict("186", "302")

    def test_predict_strips_whitespace_from_ids(self, test_client):
        payload = {"user_id": " 196 ", "movie_id": " 242 "}
        data = test_client.post("/predict", json=payload).json()
        assert (data["user_id"], data["movie_id"]) == ("196", "242")

    def test_predict_unknown_user_still_returns_valid_rating(self, test_client):
        response = test_client.post("/predict", json={"user_id": "99999", "movie_id": "242"})
        assert response.status_code == 200
        assert 1.0 <= response.json()["predicted_rating"] <= 5.0

    def test_predict_missing_user_id_returns_422(self, test_client):
        response = test_client.post("/predict", json={"movie_id": "242"})
        assert response.status_code == 422

    def test_predict_missing_movie_id_returns_422(self, test_client):
        response = test_client.post("/predict", json={"user_id": "196"})
        assert response.status_code == 422

    def test_predict_empty_body_returns_422(self, test_client):
        response = test_client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json_returns_422(self, test_client):
        response = test_client.post(
            "/predict",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_predict_rejects_invalid_requests(self, test_client, invalid_prediction_requests):
        for payload in invalid_prediction_requests:
            response = test_client.post("/predict", json=payload)
            assert response.status_code == 422, payload

    def test_predict_rejects_numeric_ids(self, test_client):
        response = test_client.post("/predict", json={"user_id": 196, "movie_id": 242})
        assert response.status_code == 422

    def test_predict_multiple_valid_requests(self, test_client, known_user_movie_pairs):
        for pair in known_user_movie_pairs:
            payload = {"user_id": pair["user_id"], "movie_id": pair["movie_id"]}
            assert test_client.post("/predict", json=payload).status_code == 200

    def test_predict_returns_503_when_model_is_missing(
        self, test_client, sample_prediction_request, monkeypatch
    ):
        monkeypatch.setattr(main, "model", None)
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 503
        assert response.json()["detail"] == "Model not loaded"

    def test_predict_returns_500_when_inference_fails(
        self, test_client, sample_prediction_request, monkeypatch
    ):
        monkeypatch.setattr(main, "model", BrokenModel())
        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 500
        assert response.json()["detail"] == "inference failed"


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, test_client, sample_batch_request):
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 200

    def test_batch_predict_returns_correct_count(self, test_client, sample_batch_request):
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert data["total_count"] == len(sample_batch_request["predictions"])
        assert len(data["predictions"]) == data["total_count"]

    def test_batch_predict_all_ratings_in_range(self, test_client, sample_batch_request):
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        assert all(1.0 <= item["predicted_rating"] <= 5.0 for item in data["predictions"])

    def test_batch_predict_preserves_request_order(self, test_client, sample_batch_request):
        data = test_client.post("/predict/batch", json=sample_batch_request).json()
        returned = [(item["user_id"], item["movie_id"]) for item in data["predictions"]]
        requested = [
            (item["user_id"], item["movie_id"]) for item in sample_batch_request["predictions"]
        ]
        assert returned == requested

    def test_batch_predict_matches_single_endpoint(self, test_client, sample_batch_request):
        batch = test_client.post("/predict/batch", json=sample_batch_request).json()
        for item in batch["predictions"]:
            single = test_client.post(
                "/predict", json={"user_id": item["user_id"], "movie_id": item["movie_id"]}
            ).json()
            assert single["predicted_rating"] == item["predicted_rating"]

    def test_batch_predict_empty_list_returns_422(self, test_client):
        response = test_client.post("/predict/batch", json={"predictions": []})
        assert response.status_code == 422

    def test_batch_predict_oversized_list_returns_422(self, test_client):
        items = [{"user_id": "1", "movie_id": str(index + 1)} for index in range(101)]
        response = test_client.post("/predict/batch", json={"predictions": items})
        assert response.status_code == 422

    def test_batch_predict_invalid_item_returns_422(self, test_client):
        response = test_client.post("/predict/batch", json={"predictions": [{"user_id": "196"}]})
        assert response.status_code == 422

    def test_batch_predict_returns_503_when_model_is_missing(
        self, test_client, sample_batch_request, monkeypatch
    ):
        monkeypatch.setattr(main, "model", None)
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 503

    def test_batch_predict_returns_500_when_inference_fails(
        self, test_client, sample_batch_request, monkeypatch
    ):
        monkeypatch.setattr(main, "model", BrokenModel())
        response = test_client.post("/predict/batch", json=sample_batch_request)
        assert response.status_code == 500


class TestErrorHandling:
    def test_404_for_unknown_endpoint(self, test_client):
        response = test_client.get("/unknown")
        assert response.status_code == 404

    def test_method_not_allowed_get_predict(self, test_client):
        response = test_client.get("/predict")
        assert response.status_code == 405

    def test_method_not_allowed_post_health(self, test_client):
        response = test_client.post("/health")
        assert response.status_code == 405

    def test_large_payload_rejected(self, test_client):
        response = test_client.post("/predict", json={"user_id": "1" * 10000, "movie_id": "242"})
        assert response.status_code in [400, 422]


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, test_client):
        response = test_client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_version(self, test_client):
        assert test_client.get("/model/info").json()["model_version"] == MODEL_VERSION

    def test_model_info_has_is_loaded(self, test_client):
        assert test_client.get("/model/info").json()["is_loaded"] is True

    def test_model_info_reports_not_loaded_without_model(self, test_client, monkeypatch):
        monkeypatch.setattr(main, "model", None)
        assert test_client.get("/model/info").json()["is_loaded"] is False


class TestStartup:
    def test_startup_survives_missing_model_file(self, monkeypatch):
        def fail_to_load():
            raise FileNotFoundError("svd_model.pkl")

        monkeypatch.setattr(main, "model", None)
        monkeypatch.setattr(main, "MovieRatingModel", fail_to_load)
        with TestClient(main.app) as client:
            assert client.get("/health").json()["status"] == "unhealthy"
            assert client.get("/").status_code == 200

    def test_startup_loads_model(self, trained_model, monkeypatch):
        monkeypatch.setattr(main, "model", None)
        with TestClient(main.app) as client:
            assert client.get("/health").json()["model_loaded"] is True
