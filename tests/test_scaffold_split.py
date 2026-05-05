"""
Tests for scaffold splitting utilities.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

rdkit = pytest.importorskip("rdkit")

from src.evaluation.scaffold_split import (
    get_murcko_scaffold,
    get_scaffold_groups,
    ScaffoldSplitter,
)


class TestGetMurckoScaffold:
    def test_benzene_derivatives_share_scaffold(self):
        """Toluene and phenol should share the benzene scaffold."""
        s1 = get_murcko_scaffold('c1ccccc1C')    # Toluene
        s2 = get_murcko_scaffold('c1ccc(O)cc1')  # Phenol
        assert s1 == s2  # Both should be benzene ring

    def test_acyclic_returns_empty_or_none(self):
        """Acyclic molecules have empty/no Murcko scaffold."""
        scaffold = get_murcko_scaffold('CCO')  # Ethanol
        # RDKit returns empty string for acyclic molecules
        assert scaffold is not None

    def test_invalid_smiles(self):
        scaffold = get_murcko_scaffold('NOT_A_MOLECULE')
        assert scaffold is None

    def test_generic_scaffold(self):
        scaffold_specific = get_murcko_scaffold('c1ccccc1', generic=False)
        scaffold_generic = get_murcko_scaffold('c1ccccc1', generic=True)
        # Both should return something for benzene
        assert scaffold_specific is not None
        assert scaffold_generic is not None


class TestScaffoldGroups:
    def test_returns_same_length(self):
        smiles = ['CCO', 'c1ccccc1', 'CCCO']
        groups = get_scaffold_groups(smiles)
        assert len(groups) == len(smiles)


class TestScaffoldSplitter:
    def test_split_covers_all_indices(self):
        smiles = ['c1ccccc1', 'c1ccccc1C', 'CCO', 'CCCO', 'c1ccc(O)cc1',
                   'c1ccc(N)cc1', 'CCCCO', 'CC(=O)O']
        splitter = ScaffoldSplitter()
        train_idx, test_idx = splitter.split(smiles, test_size=0.3)
        all_idx = set(train_idx.tolist()) | set(test_idx.tolist())
        assert all_idx == set(range(len(smiles)))

    def test_no_overlap(self):
        smiles = ['c1ccccc1', 'c1ccccc1C', 'CCO', 'CCCO', 'c1ccc(O)cc1',
                   'c1ccc(N)cc1', 'CCCCO', 'CC(=O)O']
        splitter = ScaffoldSplitter()
        train_idx, test_idx = splitter.split(smiles, test_size=0.3)
        overlap = set(train_idx.tolist()) & set(test_idx.tolist())
        assert len(overlap) == 0

    def test_deterministic(self):
        smiles = ['c1ccccc1', 'c1ccccc1C', 'CCO', 'CCCO', 'c1ccc(O)cc1']
        splitter = ScaffoldSplitter()
        train1, test1 = splitter.split(smiles, random_state=42)
        train2, test2 = splitter.split(smiles, random_state=42)
        np.testing.assert_array_equal(sorted(train1), sorted(train2))
        np.testing.assert_array_equal(sorted(test1), sorted(test2))
