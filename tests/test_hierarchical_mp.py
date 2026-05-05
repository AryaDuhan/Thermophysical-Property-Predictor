"""
Tests for the HierarchicalMP prediction pipeline.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

rdkit = pytest.importorskip("rdkit")

from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7, PredictionResult


# Small dataset for fast tests
TRAIN_SMILES = ['c1ccccc1', 'c1ccccc1C', 'CCO', 'CCCO', 'c1ccc(O)cc1',
                'c1ccc(N)cc1', 'CCCCO', 'CC(=O)O', 'CC(=O)C', 'CCCCCO']
TRAIN_TMS = np.array([278.7, 178.0, 159.0, 147.0, 316.0,
                      267.0, 183.0, 289.8, 178.5, 194.0], dtype=np.float32)


@pytest.fixture
def fitted_predictor():
    """Create a predictor fitted with sample data."""
    pred = HierarchicalMPPredictorV7()
    pred.fit_index(TRAIN_SMILES, TRAIN_TMS)
    return pred


class TestFitIndex:
    def test_fit_sets_flag(self, fitted_predictor):
        assert fitted_predictor.index_fitted is True

    def test_exact_lookup_populated(self, fitted_predictor):
        assert len(fitted_predictor.exact_lookup) > 0

    def test_stores_tm_values(self, fitted_predictor):
        assert fitted_predictor.tm_values is not None
        assert len(fitted_predictor.tm_values) > 0


class TestPredict:
    def test_exact_match(self, fitted_predictor):
        result = fitted_predictor.predict('CCO')
        assert result.method == 'exact_smiles'
        assert abs(result.tm_pred - 159.0) < 1.0
        assert result.from_cache is True

    def test_exact_match_alternate_smiles(self, fitted_predictor):
        """Same molecule in different SMILES notation should still match."""
        result = fitted_predictor.predict('OCC')  # Ethanol, different notation
        assert result.method == 'exact_smiles'
        assert abs(result.tm_pred - 159.0) < 1.0

    def test_returns_prediction_result(self, fitted_predictor):
        result = fitted_predictor.predict('CCCCCCO')
        assert isinstance(result, PredictionResult)
        assert hasattr(result, 'tm_pred')
        assert hasattr(result, 'method')
        assert hasattr(result, 'confidence')

    def test_invalid_smiles_returns_default(self, fitted_predictor):
        result = fitted_predictor.predict('INVALID_NOT_A_MOLECULE')
        assert result.method == 'default'
        assert result.confidence < 0.5

    def test_prediction_is_reasonable(self, fitted_predictor):
        """Prediction should be within a physically reasonable range for melting points."""
        result = fitted_predictor.predict('c1ccc(CC)cc1')  # Ethylbenzene-like
        assert 50 < result.tm_pred < 600  # Kelvin

    def test_interval_bounds(self, fitted_predictor):
        result = fitted_predictor.predict('CCO')
        assert result.tm_low < result.tm_pred
        assert result.tm_high > result.tm_pred


class TestBatchPredict:
    def test_batch_returns_dataframe(self, fitted_predictor):
        df = fitted_predictor.predict_batch(['CCO', 'c1ccccc1', 'CCCCCCO'])
        assert len(df) == 3
        assert 'SMILES' in df.columns
        assert 'Tm_pred' in df.columns
        assert 'method' in df.columns


class TestCalibration:
    def test_calibration_after_index(self, fitted_predictor):
        calib_smiles = ['c1ccccc1CC', 'CCCCO', 'CC(C)O']
        calib_tms = np.array([178.2, 183.0, 185.0])
        fitted_predictor.fit_calibration(calib_smiles, calib_tms)
        assert fitted_predictor.calibration_fitted is True

    def test_calibration_before_index_raises(self):
        pred = HierarchicalMPPredictorV7()
        with pytest.raises(RuntimeError):
            pred.fit_calibration(['CCO'], np.array([159.0]))

    def test_predict_before_index_raises(self):
        pred = HierarchicalMPPredictorV7()
        with pytest.raises(RuntimeError):
            pred.predict('CCO')
