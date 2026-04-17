"""
Tests for the training pipeline.

Run: pytest tests/ -v
"""

import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train.train import generate_dataset, train  # noqa: E402


class TestGenerateDataset:
    def test_creates_csv(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            df = generate_dataset(path, n=50)
            assert os.path.exists(path)
            assert len(df) == 50
        finally:
            os.unlink(path)

    def test_columns(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            df = generate_dataset(path, n=50)
            assert set(df.columns) == {"age", "income_k", "tenure_years", "target"}
        finally:
            os.unlink(path)

    def test_reproducibility(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f1:
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f2:
            path2 = f2.name
        try:
            df1 = generate_dataset(path1, n=50, seed=99)
            df2 = generate_dataset(path2, n=50, seed=99)
            pd.testing.assert_frame_equal(df1, df2)
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestTrain:
    @pytest.fixture
    def df(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        df = generate_dataset(path, n=100)
        os.unlink(path)
        return df

    def test_returns_three_items(self, df):
        result = train(df)
        assert len(result) == 3

    def test_metrics_present(self, df):
        _, _, metrics = train(df)
        for key in ("mae", "rmse", "r2"):
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_r2_is_positive(self, df):
        """LinearRegression on a linear dataset should achieve a high R²."""
        _, _, metrics = train(df)
        assert metrics["r2"] > 0.8, f"R² unexpectedly low: {metrics['r2']}"

    def test_params_present(self, df):
        _, params, _ = train(df)
        for key in ("model_type", "features", "coef_age", "coef_income_k", "coef_tenure_years"):
            assert key in params

    def test_coef_signs(self, df):
        """All features have positive coefficients by design (see data generation)."""
        _, params, _ = train(df)
        assert params["coef_age"] > 0
        assert params["coef_income_k"] > 0
        assert params["coef_tenure_years"] > 0
