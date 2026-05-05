"""
Tests for fingerprint utilities: uint64 conversion, popcount, Tanimoto similarity.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.models.hierarchical_mp_v7 import (
    fp_to_uint64_blocks,
    popcount_u64,
    fast_tanimoto_u64,
    tanimoto_single_u64,
)


def _get_fp(smiles: str):
    """Helper to get Morgan fingerprint from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


class TestFpToUint64Blocks:
    def test_output_shape(self):
        fp = _get_fp('CCO')
        blocks = fp_to_uint64_blocks(fp)
        # 2048 bits / 64 bits per uint64 = 32 blocks
        assert blocks.shape == (32,)
        assert blocks.dtype == np.uint64

    def test_different_molecules_differ(self):
        blocks1 = fp_to_uint64_blocks(_get_fp('CCO'))
        blocks2 = fp_to_uint64_blocks(_get_fp('c1ccccc1'))
        assert not np.array_equal(blocks1, blocks2)

    def test_same_molecule_same_blocks(self):
        blocks1 = fp_to_uint64_blocks(_get_fp('CCO'))
        blocks2 = fp_to_uint64_blocks(_get_fp('OCC'))  # same molecule
        assert np.array_equal(blocks1, blocks2)


class TestPopcount:
    def test_known_values(self):
        # Single uint64 with known bit count
        arr = np.array([0], dtype=np.uint64)
        assert popcount_u64(arr).sum() == 0

        arr = np.array([1], dtype=np.uint64)
        assert popcount_u64(arr).sum() == 1

        arr = np.array([0xFF], dtype=np.uint64)  # 8 bits set
        assert popcount_u64(arr).sum() == 8

    def test_2d_input(self):
        arr = np.array([[1, 3], [7, 15]], dtype=np.uint64)
        result = popcount_u64(arr)
        assert result.shape == (2, 2)
        # 1->1bit, 3->2bits, 7->3bits, 15->4bits
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


class TestTanimoto:
    def test_identical_molecules(self):
        fp = _get_fp('CCO')
        q = fp_to_uint64_blocks(fp)
        db = q.reshape(1, -1)
        sims = fast_tanimoto_u64(q, db)
        assert abs(sims[0] - 1.0) < 1e-6

    def test_different_molecules_less_than_one(self):
        q = fp_to_uint64_blocks(_get_fp('CCO'))
        d = fp_to_uint64_blocks(_get_fp('c1ccccc1'))
        db = d.reshape(1, -1)
        sims = fast_tanimoto_u64(q, db)
        assert 0.0 <= sims[0] < 1.0

    def test_single_pair_matches_batch(self):
        q = fp_to_uint64_blocks(_get_fp('CCO'))
        d = fp_to_uint64_blocks(_get_fp('CCCO'))
        # Single pair
        single = tanimoto_single_u64(q, d)
        # Batch
        batch = fast_tanimoto_u64(q, d.reshape(1, -1))
        assert abs(single - batch[0]) < 1e-6

    def test_similar_molecules_higher_score(self):
        q = fp_to_uint64_blocks(_get_fp('CCO'))       # Ethanol
        d_similar = fp_to_uint64_blocks(_get_fp('CCCO'))  # Propanol
        d_diff = fp_to_uint64_blocks(_get_fp('c1ccccc1'))  # Benzene

        sim_similar = tanimoto_single_u64(q, d_similar)
        sim_diff = tanimoto_single_u64(q, d_diff)
        # Propanol should be more similar to ethanol than benzene is
        assert sim_similar > sim_diff

    def test_symmetry(self):
        q = fp_to_uint64_blocks(_get_fp('CCO'))
        d = fp_to_uint64_blocks(_get_fp('c1ccccc1'))
        assert abs(tanimoto_single_u64(q, d) - tanimoto_single_u64(d, q)) < 1e-6
