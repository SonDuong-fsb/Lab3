import numpy as np
import pytest

from app.config import MAX_RATING, MIN_RATING

SAMPLE_SIZE = 1000
SAMPLE_SEED = 42
MAX_SAMPLE_MAE = 0.75
MAX_SAMPLE_RMSE = 1.0
MIN_PREFERENCE_GAP = 1.0


def errors_against_actuals(model, pairs):
    return np.array(
        [model.predict(p["user_id"], p["movie_id"]) - p["actual_rating"] for p in pairs]
    )


def mean_prediction(model, frame):
    return np.mean([model.predict(row.user_id, row.movie_id) for row in frame.itertuples()])


class TestModelInvariance:
    def test_same_input_same_output(self, trained_model):
        result1 = trained_model.predict("196", "242")
        result2 = trained_model.predict("196", "242")
        assert result1 == result2

    def test_multiple_calls_consistent(self, trained_model):
        results = [trained_model.predict("196", "242") for _ in range(5)]
        assert all(result == results[0] for result in results)

    def test_batch_order_independent(self, trained_model):
        pairs1 = [("196", "242"), ("186", "302")]
        pairs2 = [("186", "302"), ("196", "242")]
        results1 = trained_model.predict_batch(pairs1)
        results2 = trained_model.predict_batch(pairs2)
        assert set(results1) == set(results2)
        assert results1 == list(reversed(results2))

    def test_individual_vs_batch_same_results(self, trained_model, known_user_movie_pairs):
        pairs = [(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs]
        individual = [trained_model.predict(user, movie) for user, movie in pairs]
        assert trained_model.predict_batch(pairs) == individual

    def test_surrounding_context_does_not_change_prediction(self, trained_model):
        alone = trained_model.predict("196", "242")
        in_batch = trained_model.predict_batch([("186", "302"), ("196", "242"), ("22", "377")])
        assert in_batch[1] == alone

    def test_unknown_users_are_interchangeable(self, trained_model, unknown_users):
        predictions = {trained_model.predict(user, "242") for user in unknown_users}
        assert len(predictions) == 1


class TestModelDirectional:
    def test_predictions_are_reasonable(self, trained_model, known_user_movie_pairs):
        errors = errors_against_actuals(trained_model, known_user_movie_pairs)
        assert np.abs(errors).mean() < 1.5

    def test_different_movies_different_predictions(self, trained_model):
        predictions = {trained_model.predict("196", movie) for movie in ("242", "302", "377", "51")}
        assert len(predictions) > 1

    def test_different_users_different_predictions(self, trained_model):
        predictions = {trained_model.predict(user, "242") for user in ("196", "186", "22", "244")}
        assert len(predictions) > 1

    def test_loved_movies_score_higher_than_disliked_movies(self, trained_model, movielens_ratings):
        ordered = movielens_ratings.sort_values(["user_id", "movie_id"])
        loved = ordered[ordered["rating"] == 5].head(200)
        disliked = ordered[ordered["rating"] == 1].head(200)
        gap = mean_prediction(trained_model, loved) - mean_prediction(trained_model, disliked)
        assert gap > MIN_PREFERENCE_GAP

    def test_each_rating_level_scores_higher_than_the_one_below(
        self, trained_model, movielens_ratings
    ):
        ordered = movielens_ratings.sort_values(["user_id", "movie_id"])
        means = [
            mean_prediction(trained_model, ordered[ordered["rating"] == level].head(200))
            for level in (1, 2, 3, 4, 5)
        ]
        assert means == sorted(means)


class TestMinimumFunctionality:
    def test_can_predict_for_known_user(self, trained_model):
        prediction = trained_model.predict("196", "242")
        assert prediction is not None
        assert MIN_RATING <= prediction <= MAX_RATING

    def test_known_user_movie_pair_is_close_to_actual(self, trained_model):
        prediction = trained_model.predict("196", "242")
        assert abs(prediction - 3.0) < 1.5

    def test_can_predict_for_multiple_users(self, trained_model, known_user_movie_pairs):
        for pair in known_user_movie_pairs:
            prediction = trained_model.predict(pair["user_id"], pair["movie_id"])
            assert MIN_RATING <= prediction <= MAX_RATING

    def test_predictions_not_all_same(self, trained_model, known_user_movie_pairs):
        predictions = [
            trained_model.predict(p["user_id"], p["movie_id"]) for p in known_user_movie_pairs
        ]
        assert len(set(predictions)) > 1, "All predictions are identical"

    def test_handles_unknown_user_gracefully(self, trained_model, unknown_users):
        for user_id in unknown_users:
            prediction = trained_model.predict(user_id, "242")
            assert MIN_RATING <= prediction <= MAX_RATING

    def test_handles_unknown_movie_gracefully(self, trained_model, unknown_movies):
        for movie_id in unknown_movies:
            prediction = trained_model.predict("196", movie_id)
            assert MIN_RATING <= prediction <= MAX_RATING

    def test_handles_unknown_user_and_movie_together(self, trained_model):
        prediction = trained_model.predict("99999", "99999")
        assert MIN_RATING <= prediction <= MAX_RATING

    def test_cold_start_falls_back_to_global_average(self, trained_model, movielens_ratings):
        global_mean = movielens_ratings["rating"].mean()
        prediction = trained_model.predict("99999", "99999")
        assert prediction == pytest.approx(global_mean, abs=0.05)


class TestModelPerformance:
    def test_median_error_on_known_pairs_is_acceptable(self, trained_model, known_user_movie_pairs):
        errors = errors_against_actuals(trained_model, known_user_movie_pairs)
        assert np.median(np.abs(errors)) < 1.0

    def test_no_extreme_errors(self, trained_model, known_user_movie_pairs):
        errors = errors_against_actuals(trained_model, known_user_movie_pairs)
        assert np.abs(errors).max() < 3.0

    def test_mae_on_dataset_sample_below_threshold(self, trained_model, movielens_ratings):
        sample = movielens_ratings.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
        predictions = [trained_model.predict(r.user_id, r.movie_id) for r in sample.itertuples()]
        mae = np.abs(np.array(predictions) - sample["rating"].to_numpy()).mean()
        assert mae < MAX_SAMPLE_MAE

    def test_rmse_on_dataset_sample_below_threshold(self, trained_model, movielens_ratings):
        sample = movielens_ratings.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
        predictions = [trained_model.predict(r.user_id, r.movie_id) for r in sample.itertuples()]
        rmse = np.sqrt(np.mean((np.array(predictions) - sample["rating"].to_numpy()) ** 2))
        assert rmse < MAX_SAMPLE_RMSE

    def test_prediction_spread_is_not_degenerate(self, trained_model, movielens_ratings):
        sample = movielens_ratings.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
        predictions = [trained_model.predict(r.user_id, r.movie_id) for r in sample.itertuples()]
        assert np.std(predictions) > 0.3


class TestModelRobustness:
    def test_handles_string_numeric_ids(self, trained_model):
        prediction = trained_model.predict("196", "242")
        assert isinstance(prediction, float)

    def test_numeric_ids_are_not_matched_to_known_users(self, trained_model):
        known = trained_model.predict("196", "242")
        as_integer = trained_model.predict(196, 242)
        assert as_integer != known

    def test_handles_leading_zeros_in_ids(self, trained_model):
        padded = trained_model.predict("0196", "242")
        unknown = trained_model.predict("99999", "242")
        assert padded == unknown

    def test_handles_very_long_ids(self, trained_model):
        prediction = trained_model.predict("9" * 500, "242")
        assert MIN_RATING <= prediction <= MAX_RATING

    def test_handles_unicode_ids(self, trained_model):
        prediction = trained_model.predict("用户", "電影")
        assert MIN_RATING <= prediction <= MAX_RATING
