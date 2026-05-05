"""
Command-line interface for HierarchicalMP Predictor.

Usage:
    python -m src.cli predict CCO --format table
    python -m src.cli predict-batch input.csv -o results.csv
    python -m src.cli info
"""
import argparse
import sys
import json
import csv
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def predict_single(args):
    """Predict melting point for a single SMILES."""
    from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7
    from pathlib import Path

    model_path = Path(args.model) if args.model else None

    if model_path and model_path.exists():
        predictor = HierarchicalMPPredictorV7.load(model_path)
    else:
        print("Error: No saved model found. Provide --model path to a saved model directory.", file=sys.stderr)
        print("       To create a model, run the training pipeline first (see notebooks/).", file=sys.stderr)
        sys.exit(1)

    result = predictor.predict(args.smiles)

    if args.format == 'json':
        output = {
            'smiles': result.smiles,
            'prediction_K': round(result.tm_pred, 2),
            'prediction_C': round(result.tm_pred - 273.15, 2),
            'interval_low_K': round(result.tm_low, 2),
            'interval_high_K': round(result.tm_high, 2),
            'method': result.method,
            'confidence': round(result.confidence, 4),
            'top_similarity': round(result.top_similarity, 4),
        }
        print(json.dumps(output, indent=2))
    else:
        print("+----------------------------------------------+")
        print("|  HierarchicalMP Prediction                   |")
        print("+----------------------------------------------+")
        print(f"|  SMILES     : {result.smiles}")
        print(f"|  Prediction : {result.tm_pred:.2f} K ({result.tm_pred - 273.15:.2f} C)")
        print(f"|  Interval   : [{result.tm_low:.1f}, {result.tm_high:.1f}] K")
        print(f"|  Method     : {result.method}")
        print(f"|  Confidence : {result.confidence:.4f}")
        print("+----------------------------------------------+")


def predict_batch(args):
    """Batch predict from CSV file."""
    from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7
    from pathlib import Path

    model_path = Path(args.model) if args.model else None

    if not model_path or not model_path.exists():
        print("Error: No saved model found. Provide --model path.", file=sys.stderr)
        sys.exit(1)

    predictor = HierarchicalMPPredictorV7.load(model_path)

    # Read input
    smiles_list = []
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        smiles_col = 0
        if header:
            for i, col in enumerate(header):
                if col.upper() in ('SMILES', 'SMI', 'MOLECULE'):
                    smiles_col = i
                    break
        for row in reader:
            if row:
                smiles_list.append(row[smiles_col])

    print(f"Predicting {len(smiles_list)} molecules...")
    df = predictor.predict_batch(smiles_list)

    output = args.output or args.input.replace('.csv', '_predictions.csv')
    df.to_csv(output, index=False)
    print(f"Results saved to {output}")


def show_info(args):
    """Show model information."""
    print("+----------------------------------------------+")
    print("|  HierarchicalMP Predictor                    |")
    print("+----------------------------------------------+")
    print("|  Version    : v7.0 (Production)              |")
    print("|  Algorithm  : Hierarchical Retrieval + CNU   |")
    print("|  Coverage   : 96.8% exact match              |")
    print("|  Throughput : 948 mol/s                      |")
    print("|  Memory     : ~92 MB                         |")
    print("+----------------------------------------------+")
    print("|  Prediction Hierarchy:                       |")
    print("|    1. Exact SMILES lookup (96.8%)             |")
    print("|    2. Near-exact (Tanimoto >= 0.95)           |")
    print("|    3. Retrieval (Tanimoto >= 0.70)            |")
    print("|    4. Fallback (LightGBM + RDKit)             |")
    print("+----------------------------------------------+")


def main():
    parser = argparse.ArgumentParser(
        prog='thermophysical',
        description='HierarchicalMP: Molecular Melting Point Prediction',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # predict command
    predict_parser = subparsers.add_parser('predict', help='Predict melting point for a SMILES')
    predict_parser.add_argument('smiles', type=str, help='SMILES string of the molecule')
    predict_parser.add_argument('--model', type=str, default=None, help='Path to saved model directory')
    predict_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    predict_parser.set_defaults(func=predict_single)

    # predict-batch command
    batch_parser = subparsers.add_parser('predict-batch', help='Batch predict from CSV')
    batch_parser.add_argument('input', type=str, help='Input CSV file with SMILES column')
    batch_parser.add_argument('-o', '--output', type=str, default=None, help='Output CSV file')
    batch_parser.add_argument('--model', type=str, default=None, help='Path to saved model directory')
    batch_parser.set_defaults(func=predict_batch)

    # info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.set_defaults(func=show_info)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == '__main__':
    main()
