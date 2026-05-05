"""
Tests for the CNU uncertainty functional and calibrator.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.calibration.uncertainty_functional import (
    NeighborhoodUncertainty,
    NeighborhoodFeatures,
    learn_weights,
    compute_global_sigma,
)
from src.calibration.cnu_calibrator import CNUCalibrator, CalibrationResult


class TestNeighborhoodUncertainty:
    def test_perfect_match_low_uncertainty(self):
        nu = NeighborhoodUncertainty(global_sigma_fallback=30.0)
        sims = np.array([1.0, 0.9, 0.8])
        vals = np.array([300.0, 300.0, 300.0])
        feat = nu.compute_features(sims, vals)
        assert feat.s1 == 1.0
        assert feat.sigma_w < 1.0  # All same value → near-zero variance

    def test_no_neighbors_max_uncertainty(self):
        nu = NeighborhoodUncertainty(global_sigma_fallback=30.0)
        feat = nu.compute_features(np.array([]), np.array([]))
        score = nu.score(feat)
        assert score > 0  # Should have high uncertainty
        assert feat.s1 == 0.0

    def test_higher_variance_means_higher_uncertainty(self):
        nu = NeighborhoodUncertainty(
            weights=np.array([1.0, 1.0, 1.0, 1.0]),
            global_sigma_fallback=30.0
        )

        # Low variance neighbors
        sims = np.array([0.8, 0.75, 0.7])
        vals_low = np.array([300.0, 301.0, 299.0])
        feat_low = nu.compute_features(sims, vals_low)

        # High variance neighbors
        vals_high = np.array([300.0, 400.0, 200.0])
        feat_high = nu.compute_features(sims, vals_high)

        assert nu.score(feat_high) > nu.score(feat_low)

    def test_feature_vector_shape(self):
        nu = NeighborhoodUncertainty(global_sigma_fallback=30.0)
        sims = np.array([0.9, 0.8])
        vals = np.array([300.0, 310.0])
        feat = nu.compute_features(sims, vals)
        assert feat.z.shape == (4,)

    def test_set_weights(self):
        nu = NeighborhoodUncertainty()
        new_w = np.array([2.0, 3.0, 1.0, 0.5])
        nu.set_weights(new_w)
        np.testing.assert_array_equal(nu.weights, new_w)

    def test_set_weights_rejects_negative(self):
        nu = NeighborhoodUncertainty()
        with pytest.raises(AssertionError):
            nu.set_weights(np.array([-1.0, 1.0, 1.0, 1.0]))


class TestLearnWeights:
    def test_returns_nonnegative(self):
        # Synthetic features and residuals
        features = [
            NeighborhoodFeatures(s1=0.9, delta_s=0.1, sigma_w=5.0, k_eff=3.0,
                                 ambiguity=0.5, z=np.array([0.1, 5.0, 0.33, 0.5])),
            NeighborhoodFeatures(s1=0.5, delta_s=0.05, sigma_w=30.0, k_eff=1.5,
                                 ambiguity=2.0, z=np.array([0.5, 30.0, 0.67, 2.0])),
            NeighborhoodFeatures(s1=0.3, delta_s=0.01, sigma_w=50.0, k_eff=1.0,
                                 ambiguity=3.0, z=np.array([0.7, 50.0, 1.0, 3.0])),
        ]
        residuals = np.array([2.0, 25.0, 60.0])
        weights = learn_weights(features, residuals)
        assert len(weights) == 4
        assert np.all(weights >= 0)

    def test_weights_shape(self):
        features = [
            NeighborhoodFeatures(s1=0.8, delta_s=0.1, sigma_w=10.0, k_eff=2.0,
                                 ambiguity=1.0, z=np.array([0.2, 10.0, 0.5, 1.0]))
            for _ in range(20)
        ]
        residuals = np.random.rand(20) * 50
        weights = learn_weights(features, residuals)
        assert weights.shape == (4,)


class TestComputeGlobalSigma:
    def test_basic(self):
        values = [
            np.array([300.0, 310.0, 320.0]),
            np.array([200.0, 250.0]),
            np.array([400.0]),  # Too few, should be skipped
        ]
        sigma = compute_global_sigma(values)
        assert sigma > 0

    def test_empty_returns_default(self):
        sigma = compute_global_sigma([])
        assert sigma == 30.0


class TestCNUCalibrator:
    def _make_synthetic_data(self, n=100):
        """Create synthetic calibration data."""
        np.random.seed(42)
        predictions = np.random.normal(300, 50, n)
        actuals = predictions + np.random.normal(0, 20, n)
        features = []
        for i in range(n):
            s1 = np.random.uniform(0.3, 1.0)
            z = np.array([1 - s1, np.random.uniform(5, 50),
                          1.0 / max(s1 * 3, 1), np.random.uniform(0.1, 3.0)])
            features.append(NeighborhoodFeatures(
                s1=s1, delta_s=np.random.uniform(0, 0.3),
                sigma_w=z[1], k_eff=s1 * 3,
                ambiguity=z[3], z=z
            ))
        return predictions, actuals, features

    def test_fit_returns_calibration_result(self):
        cal = CNUCalibrator(alpha=0.1, n_regimes=3)
        preds, actuals, feats = self._make_synthetic_data()
        result = cal.fit(preds, actuals, feats)
        assert isinstance(result, CalibrationResult)
        assert cal.is_fitted

    def test_get_interval_after_fit(self):
        cal = CNUCalibrator(alpha=0.1, n_regimes=3)
        preds, actuals, feats = self._make_synthetic_data()
        cal.fit(preds, actuals, feats)

        low, high, regime = cal.get_interval(feats[0], preds[0])
        assert low < high
        assert 'regime' in regime

    def test_coverage_at_least_target(self):
        """Empirical coverage should be close to 1-alpha on calibration set."""
        alpha = 0.1
        cal = CNUCalibrator(alpha=alpha, n_regimes=5)
        preds, actuals, feats = self._make_synthetic_data(n=200)
        cal.fit(preds, actuals, feats)

        covered = 0
        for i in range(len(preds)):
            low, high, _ = cal.get_interval(feats[i], preds[i])
            if low <= actuals[i] <= high:
                covered += 1
        coverage = covered / len(preds)
        # Should be at least close to 1-alpha (with some finite-sample tolerance)
        assert coverage >= (1 - alpha) - 0.05
