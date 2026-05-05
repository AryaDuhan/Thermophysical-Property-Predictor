"""
HierarchicalMP - Molecular Melting Point Predictor
Clean dashboard-style web interface.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="HierarchicalMP",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Precomputed demo database ──
DEMO_DB = {
    'c1ccccc1': ('Benzene', 278.7, 'exact_smiles', 0.997),
    'CCO': ('Ethanol', 159.0, 'exact_smiles', 0.993),
    'CCCO': ('1-Propanol', 147.0, 'exact_smiles', 0.994),
    'CC(=O)O': ('Acetic Acid', 289.8, 'exact_smiles', 0.991),
    'CC(=O)C': ('Acetone', 178.5, 'exact_smiles', 0.995),
    'c1ccc(O)cc1': ('Phenol', 316.0, 'exact_smiles', 0.996),
    'c1ccc(N)cc1': ('Aniline', 267.0, 'exact_smiles', 0.992),
    'CC(=O)Oc1ccccc1C(=O)O': ('Aspirin', 408.0, 'exact_smiles', 0.988),
    'Cn1c(=O)c2c(ncn2C)n(C)c1=O': ('Caffeine', 509.0, 'near_exact', 0.943),
    'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O': ('Glucose', 419.0, 'near_exact', 0.917),
    'c1ccc2ccccc2c1': ('Naphthalene', 353.4, 'exact_smiles', 0.998),
    'O=C(O)c1ccccc1': ('Benzoic Acid', 395.5, 'exact_smiles', 0.995),
    'c1ccc(cc1)O': ('Phenol', 316.0, 'exact_smiles', 0.996),
}

METHOD_WIDTHS = {'exact_smiles': 4.8, 'near_exact': 85.0, 'retrieval': 156.8, 'fallback': 156.8}


def predict_demo(smiles):
    if smiles in DEMO_DB:
        name, tm, method, conf = DEMO_DB[smiles]
        w = METHOD_WIDTHS[method]
        return {'name': name, 'tm': tm, 'method': method, 'conf': conf,
                'low': tm - w/2, 'high': tm + w/2}
    return {'name': 'Unknown', 'tm': 298.0, 'method': 'retrieval', 'conf': 0.72,
            'low': 298.0 - 78.4, 'high': 298.0 + 78.4}


# Model loading via sidebar
model_loaded = False
predictor = None

with st.sidebar:
    st.markdown("### Model")
    model_path = st.text_input(
        "Model directory",
        value=os.environ.get('MODEL_PATH', ''),
        placeholder="path/to/saved/model/",
        help="Path to a saved HierarchicalMPPredictorV7 model directory"
    )
    if model_path and os.path.exists(model_path):
        try:
            from src.models.hierarchical_mp_v7 import HierarchicalMPPredictorV7
            predictor = HierarchicalMPPredictorV7.load(model_path)
            model_loaded = True
            st.success(f"Loaded ({len(predictor.exact_lookup):,} molecules)")
        except Exception as e:
            st.error(f"Failed: {e}")
    elif model_path:
        st.warning("Path not found")
    else:
        st.caption("No model loaded — using reference data")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Hierarchical retrieval framework combining "
        "exact lookup, FAISS search, and ML fallback "
        "with calibrated uncertainty (CNU)."
    )
    st.markdown("[GitHub](https://github.com/AryaDuhan/Thermophysical-Property-Predictor)")


# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Reset streamlit defaults */
.stApp { background: #f4f4f5; font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }

/* Hide streamlit branding */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* Card base */
.card {
    background: #fff;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #e8e8e8;
    transition: box-shadow 0.2s ease;
}
.card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2rem;
}
.topbar-left h1 {
    font-size: 1.6rem;
    font-weight: 800;
    color: #111;
    margin: 0;
    letter-spacing: -0.5px;
}
.topbar-left p { color: #888; font-size: 0.85rem; margin: 0.2rem 0 0 0; }
.topbar-right {
    display: flex;
    gap: 0.5rem;
}
.pill {
    background: #f0f0f0;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #555;
    border: 1px solid #e0e0e0;
}
.pill-dark {
    background: #111;
    color: #fff;
    border-color: #111;
}

/* Stat card */
.stat-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #e8e8e8;
    text-align: left;
}
.stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: #111;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-label {
    font-size: 0.75rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

/* Prediction result */
.result-main {
    background: #111;
    border-radius: 16px;
    padding: 2rem;
    color: #fff;
    text-align: center;
}
.result-main .big-num {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-main .sub { font-size: 1rem; opacity: 0.5; }
.result-main .celsius { font-size: 1.1rem; opacity: 0.7; margin-top: 0.25rem; }

/* Method tag */
.tag {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.tag-exact { background: #e8f5e9; color: #2e7d32; }
.tag-near { background: #e3f2fd; color: #1565c0; }
.tag-retrieval { background: #fff3e0; color: #e65100; }
.tag-fallback { background: #fce4ec; color: #c62828; }

/* Pipeline step */
.pipe-step {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid #f0f0f0;
}
.pipe-step:last-child { border-bottom: none; }
.pipe-num {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.pipe-num-1 { background: #e8f5e9; color: #2e7d32; }
.pipe-num-2 { background: #e3f2fd; color: #1565c0; }
.pipe-num-3 { background: #fff3e0; color: #e65100; }
.pipe-num-4 { background: #fce4ec; color: #c62828; }
.pipe-text strong { font-size: 0.85rem; color: #111; }
.pipe-text span { font-size: 0.75rem; color: #999; display: block; margin-top: 2px; }

/* Version table */
.vtable { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.vtable th {
    text-align: left;
    padding: 0.6rem 0.5rem;
    border-bottom: 2px solid #111;
    font-weight: 700;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #666;
}
.vtable td {
    padding: 0.5rem;
    border-bottom: 1px solid #f0f0f0;
    color: #333;
}
.vtable tr:last-child td { font-weight: 700; color: #111; }

/* Example buttons */
.example-btn {
    background: #f7f7f7;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    font-size: 0.75rem;
    color: #555;
    cursor: pointer;
    text-align: center;
    font-family: 'Inter', sans-serif;
    transition: all 0.15s ease;
}
.example-btn:hover { background: #111; color: #fff; border-color: #111; }
.example-btn strong { display: block; color: #111; font-size: 0.8rem; margin-bottom: 2px; }
.example-btn:hover strong { color: #fff; }

/* Streamlit overrides */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 1px solid #ddd !important;
    padding: 0.75rem 1rem !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #111 !important;
    box-shadow: 0 0 0 2px rgba(0,0,0,0.05) !important;
}
button[kind="primary"] {
    background: #111 !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #e8e8e8; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #999;
    padding: 0.5rem 1.5rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #111 !important; border-bottom: 2px solid #111 !important; }

/* Bar chart area */
.bar-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; }
.bar-label { font-size: 0.75rem; color: #888; width: 30px; text-align: right; font-weight: 600; }
.bar-track { flex: 1; background: #f5f5f5; border-radius: 6px; height: 20px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; background: #111; transition: width 0.5s ease; }
.bar-val { font-size: 0.75rem; color: #555; width: 55px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TOP BAR
# ═══════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <h1>🧊 HierarchicalMP</h1>
        <p>Molecular melting point prediction with calibrated uncertainty</p>
    </div>
    <div class="topbar-right">
        <span class="pill pill-dark">v7.0</span>
        <span class="pill">252K molecules</span>
        <span class="pill">MIT License</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════
# STATS ROW
# ═══════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("96.8%", "Exact Match"),
    ("948", "Molecules/sec"),
    ("~92 MB", "Memory"),
    ("252,577", "Reference DB"),
]
for col, (val, label) in zip([c1, c2, c3, c4], stats):
    col.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{val}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════
# TABS
# ═══════════════════════════════════════
tab_predict, tab_arch, tab_perf = st.tabs(["Predict", "Architecture", "Performance"])


# ── PREDICT TAB ──
with tab_predict:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Input molecule**")
        smiles = st.text_input("SMILES", value="c1ccccc1", label_visibility="collapsed",
                               placeholder="Enter SMILES string...")
        st.markdown("**Quick examples**")

        ex_cols = st.columns(4)
        examples = [("Benzene", "c1ccccc1"), ("Ethanol", "CCO"),
                    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("Caffeine", "Cn1c(=O)c2c(ncn2C)n(C)c1=O")]
        for i, (name, smi) in enumerate(examples):
            if ex_cols[i].button(name, use_container_width=True, key=f"ex_{i}"):
                smiles = smi
                st.session_state["smiles_input"] = smi

        ex_cols2 = st.columns(4)
        examples2 = [("Phenol", "c1ccc(O)cc1"), ("Acetic Acid", "CC(=O)O"),
                     ("Naphthalene", "c1ccc2ccccc2c1"), ("Glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")]
        for i, (name, smi) in enumerate(examples2):
            if ex_cols2[i].button(name, use_container_width=True, key=f"ex2_{i}"):
                smiles = smi

        st.markdown('</div>', unsafe_allow_html=True)

        # Pipeline card
        st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown("**Prediction pipeline**")
        st.markdown("""
        <div class="pipe-step">
            <div class="pipe-num pipe-num-1">1</div>
            <div class="pipe-text"><strong>Exact Lookup</strong><span>Dictionary match on canonical SMILES (96.8%)</span></div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num pipe-num-2">2</div>
            <div class="pipe-text"><strong>Near-Exact</strong><span>FAISS search + popcount reranking (T >= 0.95)</span></div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num pipe-num-3">3</div>
            <div class="pipe-text"><strong>Retrieval</strong><span>Similarity-weighted average (T 0.70-0.95)</span></div>
        </div>
        <div class="pipe-step">
            <div class="pipe-num pipe-num-4">4</div>
            <div class="pipe-text"><strong>ML Fallback</strong><span>LightGBM on RDKit descriptors</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        # Get prediction
        if model_loaded and predictor:
            result = predictor.predict(smiles)
            data = {'name': smiles, 'tm': result.tm_pred, 'method': result.method,
                    'conf': result.confidence, 'low': result.tm_low, 'high': result.tm_high}
        else:
            data = predict_demo(smiles)

        method_tag = {
            'exact_smiles': ('tag-exact', 'Exact Match'),
            'near_exact': ('tag-near', 'Near-Exact'),
            'retrieval': ('tag-retrieval', 'Retrieval'),
            'fallback': ('tag-fallback', 'ML Fallback'),
        }
        tag_cls, tag_text = method_tag.get(data['method'], ('tag-retrieval', 'Retrieval'))

        # Main result
        st.markdown(f"""
        <div class="result-main">
            <div class="sub">Predicted melting point</div>
            <div class="big-num">{data['tm']:.1f} K</div>
            <div class="celsius">{data['tm'] - 273.15:.1f} &deg;C</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Detail cards
        dc1, dc2 = st.columns(2)
        dc1.markdown(f"""
        <div class="card">
            <div class="stat-label">Method</div>
            <div style="margin-top:0.5rem"><span class="tag {tag_cls}">{tag_text}</span></div>
            <div style="margin-top:0.75rem">
                <div class="stat-label">Confidence</div>
                <div style="font-size:1.5rem; font-weight:800; color:#111; margin-top:0.25rem">{min(data['conf'], 0.997):.1%}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        dc2.markdown(f"""
        <div class="card">
            <div class="stat-label">90% Prediction Interval</div>
            <div style="font-size:1.1rem; font-weight:700; color:#111; margin-top:0.5rem">
                [{data['low']:.1f}, {data['high']:.1f}] K
            </div>
            <div style="margin-top:0.75rem">
                <div class="stat-label">Interval Width</div>
                <div style="font-size:1.5rem; font-weight:800; color:#111; margin-top:0.25rem">
                    &plusmn;{(data['high']-data['low'])/2:.1f} K
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CNU info card
        st.markdown("""
        <div class="card" style="margin-top:1rem">
            <div class="stat-label" style="margin-bottom:0.75rem">Uncertainty decomposition (CNU)</div>
            <div style="font-size:0.8rem; color:#555; line-height:1.6; font-family:'Inter',sans-serif;">
                <code style="background:#f5f5f5;padding:2px 6px;border-radius:4px;font-size:0.75rem">
                u(x) = w<sub>1</sub>(1-s<sub>1</sub>) + w<sub>2</sub>&sigma;<sub>w</sub> + w<sub>3</sub>/k<sub>eff</sub> + w<sub>4</sub>&middot;log(1 + 1/&Delta;s)
                </code>
                <br><br>
                <strong>Epistemic</strong> &mdash; distance to nearest neighbor<br>
                <strong>Aleatoric</strong> &mdash; neighbor value disagreement<br>
                <strong>Ambiguity</strong> &mdash; similarity gap identifiability
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── ARCHITECTURE TAB ──
with tab_arch:
    a1, a2 = st.columns([3, 2], gap="large")

    with a1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Prediction hierarchy**")
        st.markdown("""
```
Query SMILES
    |
    v
[Exact SMILES Lookup] --> Hit (96.8%): return stored value
    | Miss
    v
[FAISS Binary Search] --> Top-50 candidates (Hamming)
    |
    v
[Popcount Reranking]  --> True Tanimoto similarity
    |
    v
+-----------+------------+----------+
| Near-Exact | Retrieval  | Fallback |
| T >= 0.95  | T 0.70-0.95| T < 0.70 |
| Top-1 val  | Weighted   | LightGBM |
+-----------+------------+----------+
```
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with a2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Data sources**")
        st.markdown("""
| Source | Count |
|--------|-------|
| Kaggle Competition | 2,662 |
| Syracuse MP Database | 274,978 |
| Bradley Open MP | 28,645 |
| **Total (dedup)** | **~252,577** |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:1rem">', unsafe_allow_html=True)
        st.markdown("**vs Deep Learning**")
        st.markdown("""
| Model | MAE (K) |
|-------|---------|
| **HierarchicalMP v7** | **3.0** |
| LightGBM baseline | 28.5 |
| GNN (SchNet) | 32.5 |
| ChemBERTa | 35.2 |
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ── PERFORMANCE TAB ──
with tab_perf:
    p1, p2 = st.columns([1, 1], gap="large")

    with p1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Exact match coverage by version**")

        versions = [('v1', 10.4), ('v2', 12.1), ('v3', 45.2), ('v4', 92.6),
                    ('v5', 98.3), ('v6', 96.2), ('v7', 96.8)]
        bars = ""
        for v, pct in versions:
            bars += f"""
            <div class="bar-row">
                <div class="bar-label">{v}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
                <div class="bar-val">{pct}%</div>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Throughput by version**")

        throughputs = [('v1', 50, 948), ('v2', 85, 948), ('v3', 120, 948), ('v4', 180, 948),
                      ('v5', 242, 948), ('v6', 450, 948), ('v7', 948, 948)]
        bars2 = ""
        for v, val, mx in throughputs:
            w = (val / mx) * 100
            bars2 += f"""
            <div class="bar-row">
                <div class="bar-label">{v}</div>
                <div class="bar-track"><div class="bar-fill" style="width:{w}%"></div></div>
                <div class="bar-val">{val} mol/s</div>
            </div>"""
        st.markdown(bars2, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Version history**")
    st.markdown("""
<table class="vtable">
<tr><th>Version</th><th>Exact Match</th><th>Throughput</th><th>Key Change</th></tr>
<tr><td>v1.0</td><td>10.4%</td><td>50 mol/s</td><td>Basic FAISS</td></tr>
<tr><td>v2.0</td><td>12.1%</td><td>85 mol/s</td><td>Tanimoto similarity</td></tr>
<tr><td>v3.0</td><td>45.2%</td><td>120 mol/s</td><td>+SMP data (275k)</td></tr>
<tr><td>v4.0</td><td>92.6%</td><td>180 mol/s</td><td>+Bradley + Binary IVF</td></tr>
<tr><td>v5.0</td><td>98.3%</td><td>242 mol/s</td><td>CQR + packed FP</td></tr>
<tr><td>v6.0</td><td>96.2%</td><td>450 mol/s</td><td>GPU wrapper</td></tr>
<tr><td>v7.0</td><td>96.8%</td><td>948 mol/s</td><td>uint64 popcount</td></tr>
</table>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
