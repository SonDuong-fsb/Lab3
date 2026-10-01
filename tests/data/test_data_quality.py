import numpy as np
import pytest

from app.config import MAX_RATING, MIN_RATING

REQUIRED_FIELDS = ["user_id", "movie_id", "rating"]


class TestRatingDataQuality:
    def test_all_ratings_in_valid_range(self, sample_ratings):
        for record in sample_ratings:
            assert MIN_RATING <= record["rating"] <= MAX_RATING

    def test_no_negative_ratings(self, sample_ratings):
        assert all(record["rating"] >= 0 for record in sample_ratings)

    def test_no_ratings_above_maximum(self, sample_ratings):
        assert all(record["rating"] <= MAX_RATING for record in sample_ratings)

    def test_no_missing_user_ids(self, sample_ratings):
        for record in sample_ratings:
            assert record["user_id"] is not None
            assert record["user_id"] != ""

    def test_no_missing_movie_ids(self, sample_ratings):
        for record in sample_ratings:
            assert record["movie_id"] is not None
            assert record["movie_id"] != ""

    def test_user_ids_are_strings(self, sample_ratings):
        assert all(isinstance(record["user_id"], str) for record in sample_ratings)

    def test_movie_ids_are_strings(self, sample_ratings):
        assert all(isinstance(record["movie_id"], str) for record in sample_ratings)

    def test_no_null_ratings(self, sample_ratings):
        assert all(record["rating"] is not None for record in sample_ratings)

    def test_all_records_have_required_fields(self, sample_ratings):
        for record in sample_ratings:
            for field in REQUIRED_FIELDS:
                assert field in record


class TestRatingDistribution:
    def test_mean_rating_reasonable(self, sample_ratings):
        ratings = [record["rating"] for record in sample_ratings]
        assert 2.0 <= np.mean(ratings) <= 4.5

    def test_rating_standard_deviation(self, sample_ratings):
        ratings = [record["rating"] for record in sample_ratings]
        std = np.std(ratings)
        assert 0 < std < 2.0

    def test_multiple_rating_values_exist(self, sample_ratings):
        unique_ratings = {record["rating"] for record in sample_ratings}
        assert len(unique_ratings) > 1


class TestDataUniqueness:
    def test_unique_user_movie_combinations(self, sample_ratings):
        pairs = [(record["user_id"], record["movie_id"]) for record in sample_ratings]
        assert len(pairs) == len(set(pairs))

    def test_multiple_users_exist(self, sample_ratings):
        assert len({record["user_id"] for record in sample_ratings}) > 1

    def test_multiple_movies_exist(self, sample_ratings):
        assert len({record["movie_id"] for record in sample_ratings}) > 1


class TestDataTypes:
    def test_ratings_are_numeric(self, sample_ratings):
        assert all(isinstance(record["rating"], (int, float)) for record in sample_ratings)

    def test_ratings_are_float_or_int(self, sample_ratings):
        for record in sample_ratings:
            assert isinstance(float(record["rating"]), float)


class TestMovieLensDataset:
    def test_dataset_is_not_empty(self, movielens_ratings):
        assert len(movielens_ratings) == 100_000

    def test_schema_has_expected_columns(self, movielens_ratings):
        assert list(movielens_ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]

    def test_no_missing_values(self, movielens_ratings):
        assert movielens_ratings.isna().sum().sum() == 0

    def test_ids_are_strings_without_blanks(self, movielens_ratings):
        for column in ("user_id", "movie_id"):
            values = movielens_ratings[column]
            assert values.map(type).eq(str).all()
            assert values.str.strip().ne("").all()

    def test_ratings_are_integers_between_one_and_five(self, movielens_ratings):
        ratings = movielens_ratings["rating"]
        assert ratings.between(MIN_RATING, MAX_RATING).all()
        assert (ratings % 1 == 0).all()
        assert set(ratings.unique()) == {1, 2, 3, 4, 5}

    def test_no_duplicate_user_movie_pairs(self, movielens_ratings):
        assert not movielens_ratings.duplicated(["user_id", "movie_id"]).any()

    def test_expected_number_of_users_and_movies(self, movielens_ratings):
        assert movielens_ratings["user_id"].nunique() == 943
        assert movielens_ratings["movie_id"].nunique() == 1682

    def test_every_user_has_minimum_activity(self, movielens_ratings):
        assert movielens_ratings.groupby("user_id").size().min() >= 20

    def test_mean_rating_is_in_expected_band(self, movielens_ratings):
        assert 3.0 <= movielens_ratings["rating"].mean() <= 4.0

    def test_rating_standard_deviation_is_in_expected_band(self, movielens_ratings):
        assert 0.5 <= movielens_ratings["rating"].std() <= 1.5

    def test_no_single_rating_value_dominates(self, movielens_ratings):
        share = movielens_ratings["rating"].value_counts(normalize=True)
        assert share.max() < 0.5

    def test_timestamps_are_positive(self, movielens_ratings):
        assert (movielens_ratings["timestamp"] > 0).all()

    @pytest.mark.parametrize(
        ("user_id", "movie_id", "rating"), [("196", "242", 3), ("22", "377", 1)]
    )
    def test_known_ratings_exist(self, movielens_ratings, user_id, movie_id, rating):
        match = movielens_ratings[
            (movielens_ratings["user_id"] == user_id) & (movielens_ratings["movie_id"] == movie_id)
        ]
        assert match["rating"].tolist() == [rating]
