"""
HierarchicalMP Web Demo - Streamlit App

Interactive melting point prediction with uncertainty quantification.
Runs with or without a trained model (demo mode shows architecture).
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Page config
st.set_page_config(
    page_title="HierarchicalMP - Melting Point Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.85;
        margin: 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(168, 237, 234, 0.2);
        text-align: center;
        color: white;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 237, 234, 0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-box {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(168, 237, 234, 0.3);
        color: white;
        margin: 1rem 0;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #a8edea, #fed6e3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .method-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .method-exact { background: rgba(46, 213, 115, 0.2); color: #2ed573; border: 1px solid rgba(46, 213, 115, 0.4); }
    .method-near { background: rgba(116, 185, 255, 0.2); color: #74b9ff; border: 1px solid rgba(116, 185, 255, 0.4); }
    .method-retrieval { background: rgba(253, 203, 110, 0.2); color: #fdcb6e; border: 1px solid rgba(253, 203, 110, 0.4); }
    .method-fallback { background: rgba(214, 48, 49, 0.2); color: #ff7675; border: 1px solid rgba(214, 48, 49, 0.4); }

    .hierarchy-step {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 3px solid;
        color: white;
    }

    .step-exact { border-color: #2ed573; }
    .step-near { border-color: #74b9ff; }
    .step-retrieval { border-color: #fdcb6e; }
    .step-fallback { border-color: #ff7675; }

    .info-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>HierarchicalMP</h1>
    <p>Data-Centric Molecular Melting Point Prediction with Calibrated Uncertainty</p>
</div>
""", unsafe_allow_html=True)

# --- Key Metrics ---
cols = st.columns(4)
metrics = [
    ("96.8%", "Exact Match Coverage"),
    ("948", "Molecules / Second"),
    ("~92 MB", "Memory Footprint"),
    ("252K+", "Reference Molecules"),
]
for col, (value, label) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Settings")
    demo_mode = True

    # Try to load model
    model_loaded = False
    try:
        from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7
        model_path = st.text_input("Model path (optional)", placeholder="models/v7/")
        if model_path and os.path.exists(model_path):
            with st.spinner("Loading model..."):
                predictor = HierarchicalMPPredictorV7.load(model_path)
                model_loaded = True
                demo_mode = False
                st.success(f"Model loaded: {len(predictor.exact_lookup):,} molecules")
    except ImportError:
        st.info("RDKit not available. Running in demo mode.")

    if demo_mode:
        st.warning("Running in **demo mode** (no trained model loaded). Showing architecture and sample predictions.")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **HierarchicalMP** uses a hierarchical retrieval framework 
    combining exact lookup, FAISS similarity search, and 
    ML fallback with calibrated uncertainty quantification.
    """)
    st.markdown("---")
    st.markdown("### Links")
    st.markdown("[GitHub Repository](https://github.com/AryaDuhan/Thermophysical-Property-Predictor)")


# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["Predict", "Architecture", "Performance"])


# ========== TAB 1: Predict ==========
with tab1:
    st.markdown("### Molecular Melting Point Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        smiles_input = st.text_input(
            "Enter SMILES string",
            value="c1ccccc1",
            placeholder="e.g. CCO, c1ccccc1, CC(=O)O",
            help="Enter a valid SMILES string for melting point prediction"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    # Sample molecules
    st.markdown("**Try these examples:**")
    example_cols = st.columns(5)
    examples = [
        ("Benzene", "c1ccccc1"),
        ("Ethanol", "CCO"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ]
    for col, (name, smi) in zip(example_cols, examples):
        if col.button(name, use_container_width=True):
            smiles_input = smi

    if predict_btn or smiles_input:
        st.markdown("---")

        if model_loaded and not demo_mode:
            # Real prediction
            with st.spinner("Predicting..."):
                start = time.time()
                result = predictor.predict(smiles_input)
                elapsed = time.time() - start

            method_class = {
                'exact_smiles': 'exact',
                'near_exact': 'near',
                'retrieval': 'retrieval',
                'fallback': 'fallback',
                'default': 'fallback',
            }.get(result.method, 'fallback')

            pred_cols = st.columns([1, 1, 1])

            with pred_cols[0]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">PREDICTED MELTING POINT</div>
                    <div class="prediction-value">{result.tm_pred:.1f} K</div>
                    <div style="text-align: center; opacity: 0.6; font-size: 1rem;">{result.tm_pred - 273.15:.1f} C</div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[1]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">90% PREDICTION INTERVAL</div>
                    <div class="prediction-value" style="font-size: 1.8rem;">[{result.tm_low:.1f}, {result.tm_high:.1f}] K</div>
                    <div style="text-align: center; opacity: 0.6;">Width: {result.interval_width:.1f} K</div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[2]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">METHOD</div>
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="method-badge method-{method_class}">{result.method.replace('_', ' ')}</span>
                    </div>
                    <div style="text-align: center; opacity: 0.6;">
                        Confidence: {result.confidence:.2%} | {elapsed*1000:.0f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Demo mode — show sample results
            demo_data = {
                'c1ccccc1': {'tm': 278.7, 'method': 'exact_smiles', 'conf': 1.0, 'low': 268.7, 'high': 288.7},
                'CCO': {'tm': 159.0, 'method': 'exact_smiles', 'conf': 1.0, 'low': 149.0, 'high': 169.0},
                'CC(=O)Oc1ccccc1C(=O)O': {'tm': 408.0, 'method': 'exact_smiles', 'conf': 1.0, 'low': 398.0, 'high': 418.0},
                'Cn1c(=O)c2c(ncn2C)n(C)c1=O': {'tm': 509.0, 'method': 'exact_smiles', 'conf': 1.0, 'low': 499.0, 'high': 519.0},
                'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O': {'tm': 419.0, 'method': 'near_exact', 'conf': 0.97, 'low': 395.0, 'high': 443.0},
            }

            data = demo_data.get(smiles_input, {
                'tm': 300.0, 'method': 'retrieval', 'conf': 0.75, 'low': 260.0, 'high': 340.0
            })

            method_class = {
                'exact_smiles': 'exact',
                'near_exact': 'near',
                'retrieval': 'retrieval',
                'fallback': 'fallback',
            }.get(data['method'], 'retrieval')

            pred_cols = st.columns([1, 1, 1])

            with pred_cols[0]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">PREDICTED MELTING POINT</div>
                    <div class="prediction-value">{data['tm']:.1f} K</div>
                    <div style="text-align: center; opacity: 0.6; font-size: 1rem;">{data['tm'] - 273.15:.1f} C</div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[1]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">90% PREDICTION INTERVAL</div>
                    <div class="prediction-value" style="font-size: 1.8rem;">[{data['low']:.1f}, {data['high']:.1f}] K</div>
                    <div style="text-align: center; opacity: 0.6;">Width: {data['high'] - data['low']:.1f} K</div>
                </div>
                """, unsafe_allow_html=True)

            with pred_cols[2]:
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">METHOD</div>
                    <div style="text-align: center; margin: 1rem 0;">
                        <span class="method-badge method-{method_class}">{data['method'].replace('_', ' ')}</span>
                    </div>
                    <div style="text-align: center; opacity: 0.6;">
                        Confidence: {data['conf']:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.info("Demo mode: Showing pre-computed results. Load a trained model for real predictions.")


# ========== TAB 2: Architecture ==========
with tab2:
    st.markdown("### Prediction Hierarchy")
    st.markdown("Our framework uses a hierarchical retrieval approach, routing each query through increasingly general methods:")

    st.markdown("""
    <div class="hierarchy-step step-exact">
        <strong>1. Exact SMILES Lookup (96.8% of queries)</strong><br>
        <span style="opacity: 0.7;">Dictionary lookup of canonical SMILES. O(1) time, zero error.</span>
    </div>
    <div class="hierarchy-step step-near">
        <strong>2. Near-Exact Match (Tanimoto >= 0.95)</strong><br>
        <span style="opacity: 0.7;">FAISS binary search + popcount reranking. Returns nearest neighbor value.</span>
    </div>
    <div class="hierarchy-step step-retrieval">
        <strong>3. Similarity-Weighted Retrieval (Tanimoto 0.70-0.95)</strong><br>
        <span style="opacity: 0.7;">Top-k neighbors weighted by Tanimoto^2. Calibrated via CNU.</span>
    </div>
    <div class="hierarchy-step step-fallback">
        <strong>4. ML Fallback (Tanimoto < 0.70)</strong><br>
        <span style="opacity: 0.7;">LightGBM on RDKit descriptors, predicting residual over neighbor mean.</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Calibrated Neighborhood Uncertainty (CNU)")
    st.markdown("""
    Our key theoretical contribution is a first-principles uncertainty functional derived from retrieval geometry:
    
    **u(x) = w1(1-s1) + w2*sigma_w + w3/k_eff + w4*log(1 + 1/(delta_s + eps))**
    
    Where:
    - **(1-s1)**: Distance to nearest neighbor (epistemic uncertainty)
    - **sigma_w**: Weighted variance of neighbor values (aleatoric uncertainty)
    - **1/k_eff**: Inverse effective sample size (sparsity)
    - **log(1 + 1/delta_s)**: Ambiguity from similarity gap
    
    Weights are learned via non-negative least squares (NNLS), enforcing monotonicity.
    """)


# ========== TAB 3: Performance ==========
with tab3:
    st.markdown("### Version Evolution")

    version_data = pd.DataFrame({
        'Version': ['v1.0', 'v2.0', 'v3.0', 'v4.0', 'v5.0', 'v6.0', 'v7.0'],
        'Exact Match (%)': [10.4, 12.1, 45.2, 92.6, 98.3, 96.2, 96.8],
        'Throughput (mol/s)': [50, 85, 120, 180, 242, 450, 948],
        'Key Change': [
            'Basic FAISS',
            'Tanimoto similarity',
            '+SMP data (275k)',
            '+Bradley + Binary IVF',
            'CQR + packed FP',
            'GPU wrapper',
            'uint64 popcount',
        ],
    })

    st.dataframe(version_data, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Exact Match Coverage")
        chart_data = pd.DataFrame({
            'Version': version_data['Version'],
            'Coverage (%)': version_data['Exact Match (%)'],
        })
        st.bar_chart(chart_data.set_index('Version'))

    with col2:
        st.markdown("### Throughput")
        chart_data2 = pd.DataFrame({
            'Version': version_data['Version'],
            'mol/s': version_data['Throughput (mol/s)'],
        })
        st.bar_chart(chart_data2.set_index('Version'))

    st.markdown("### Data Sources")
    data_sources = pd.DataFrame({
        'Source': ['Kaggle Competition', 'Syracuse MP Database', 'Bradley Open MP', 'Total (deduplicated)'],
        'Molecules': ['2,662', '274,978', '28,645', '~252,577'],
        'Description': [
            'Original training data',
            'Public melting point collection',
            'Jean-Claude Bradley dataset',
            'After deduplication',
        ],
    })
    st.dataframe(data_sources, use_container_width=True, hide_index=True)

    st.markdown("### Comparison with Deep Learning")
    comparison = pd.DataFrame({
        'Approach': ['HierarchicalMP v7', 'LightGBM Baseline', 'GNN (SchNet)', 'ChemBERTa'],
        'MAE (K)': [3.0, 28.5, 32.5, 35.2],
        'Note': [
            'Exact matches (calibration set)',
            'Kaggle data only',
            '2.6k training samples',
            'Fine-tuned transformer',
        ],
    })
    st.dataframe(comparison, use_container_width=True, hide_index=True)
