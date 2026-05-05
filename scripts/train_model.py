"""
Train HierarchicalMP v7 model from competition data.

Usage:
    .venv/Scripts/python.exe scripts/train_model.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7


def main():
    # ── Load data ──
    print("Loading competition data...")
    train_df = pd.read_csv('data/raw/train.csv')[['SMILES', 'Tm']]
    print(f"  Training set: {len(train_df)} molecules")

    smiles = train_df['SMILES'].tolist()
    tms = train_df['Tm'].values.astype(np.float32)

    # ── Split: 90% train, 10% calibration ──
    np.random.seed(42)
    n = len(smiles)
    idx = np.random.permutation(n)
    split = int(0.9 * n)

    train_smiles = [smiles[i] for i in idx[:split]]
    train_tms = tms[idx[:split]]
    calib_smiles = [smiles[i] for i in idx[split:]]
    calib_tms = tms[idx[split:]]

    print(f"  Train split: {len(train_smiles)}")
    print(f"  Calib split: {len(calib_smiles)}")

    # ── Build model ──
    print("\nInitializing HierarchicalMP v7...")
    predictor = HierarchicalMPPredictorV7(alpha=0.10)

    print("Building retrieval index...")
    predictor.fit_index(train_smiles, train_tms)

    print("Fitting CNU calibration...")
    predictor.fit_calibration(calib_smiles, calib_tms)

    # ── Save model ──
    save_path = "models/v7"
    print(f"\nSaving model to '{save_path}'...")
    predictor.save(save_path)

    # ── Build SMILES→name mapping for the web app dropdown ──
    print("Building molecule name lookup...")
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    name_map = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        # Use molecular formula as a fallback name
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mw = round(Descriptors.MolWt(mol), 1)
        name_map[can] = {
            'formula': formula,
            'mw': mw,
            'original_smiles': smi,
        }

    with open(os.path.join(save_path, 'molecule_names.json'), 'w') as f:
        json.dump(name_map, f)

    print(f"  Saved {len(name_map)} molecule entries")

    # ── Quick verification ──
    print("\nVerification:")
    test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']
    for s in test_smiles:
        result = predictor.predict(s)
        print(f"  {s:20s} -> {result.tm_pred:7.1f} K  ({result.method})")

    print("\nDone! Model saved to models/v7/")


if __name__ == '__main__':
    main()
