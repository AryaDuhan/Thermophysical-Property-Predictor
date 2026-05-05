"""
Shared test fixtures for Thermophysical Property Predictor.
"""
import pytest
import numpy as np


# Simple SMILES that don't require external data
SAMPLE_SMILES = [
    'c1ccccc1',      # Benzene
    'c1ccccc1C',     # Toluene
    'c1ccccc1CC',    # Ethylbenzene
    'c1ccc(O)cc1',   # Phenol
    'c1ccc(N)cc1',   # Aniline
    'CCO',           # Ethanol
    'CCCO',          # Propanol
    'CCCCO',         # Butanol
    'CC(=O)O',       # Acetic acid
    'CC(=O)C',       # Acetone
]

SAMPLE_TMS = np.array([
    278.7, 178.0, 178.2, 316.0, 267.0,
    159.0, 147.0, 183.0, 289.8, 178.5
], dtype=np.float32)


@pytest.fixture
def sample_smiles():
    return SAMPLE_SMILES.copy()


@pytest.fixture
def sample_tms():
    return SAMPLE_TMS.copy()


@pytest.fixture
def train_test_split():
    """Provide a simple train/test split."""
    return {
        'train_smiles': SAMPLE_SMILES[:7],
        'train_tms': SAMPLE_TMS[:7],
        'test_smiles': SAMPLE_SMILES[7:],
        'test_tms': SAMPLE_TMS[7:],
    }
