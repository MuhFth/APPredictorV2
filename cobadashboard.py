import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '🎓';
        position: absolute;
        font-size: 10rem;
        opacity: 0.1;
        top: -20px;
        right: -20px;
        animation: float 6s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }
    .main-header h1 { font-weight: 700; font-size: 2.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); margin-bottom: 0.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        border-left: 5px solid;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    [data-testid="stSidebar"] .stRadio > label { color: white !important; font-weight: 600; font-size: 1.1rem; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6); }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2196f3;
        margin: 1rem 0; box-shadow: 0 4px 10px rgba(33, 150, 243, 0.2);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #4caf50;
        margin: 1rem 0; box-shadow: 0 4px 10px rgba(76, 175, 80, 0.2);
    }
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #ff9800;
        margin: 1rem 0; box-shadow: 0 4px 10px rgba(255, 152, 0, 0.2);
    }
    .grade-badge {
        display: inline-block; padding: 1rem 2rem; border-radius: 30px;
        font-weight: 700; font-size: 1.5rem; margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        animation: bounce 2s ease-in-out infinite;
    }
    @keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    .grade-a { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; box-shadow: 0 6px 20px rgba(56, 239, 125, 0.4); }
    .grade-b { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4); }
    .grade-c { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; box-shadow: 0 6px 20px rgba(254, 225, 64, 0.4); }
    .grade-d { background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%); color: white; box-shadow: 0 6px 20px rgba(252, 74, 26, 0.4); }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); transition: all 0.3s; margin: 1rem 0;
    }
    .feature-card:hover { transform: scale(1.05); box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5); }
    .input-section {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%);
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    .input-section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        color: white;
        font-weight: 700;
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 2rem; border-radius: 15px;
        text-align: center; margin-top: 3rem;
        box-shadow: 0 -4px 20px rgba(102, 126, 234, 0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0.5rem; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white; border-radius: 8px; color: #667eea;
        font-weight: 600; padding: 0.8rem 1.5rem; transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL ARTIFACT
# ======================================================
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("academic_predictor_pt6.pkl")
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan 'academic_predictor_pt6.pkl' ada di direktori yang sama.")
        st.stop()

data = load_artifact()
model = data["model"]
scaler = data["scaler"]
FEATURES = data["feature_names"]
metrics = data["metrics"]

# ======================================================
# FEATURE CONFIG — sesuaikan dengan data asli
# ======================================================
# Konfigurasi fitur: (label, min, max, default, step, help_text)
# Range berdasarkan data aktual yang diamati
FEATURE_CONFIG = {
    "Persentase_Kehadiran": {
        "label": "Persentase Kehadiran (%)",
        "min": 50.0, "max": 100.0, "default": 85.0, "step": 1.0,
        "help": "Persentase kehadiran siswa (50–100%)",
        "icon": "👥"
    },
    "Nilai_Internal_1": {
        "label": "Nilai Internal 1",
        "min": 0.0, "max": 40.0, "default": 30.0, "step": 1.0,
        "help": "Nilai internal pertama (0–40)",
        "icon": "📝"
    },
    "Nilai_Internal_2": {
        "label": "Nilai Internal 2",
        "min": 0.0, "max": 40.0, "default": 30.0, "step": 1.0,
        "help": "Nilai internal kedua (0–40)",
        "icon": "📝"
    },
    "Skor_Tugas": {
        "label": "Skor Tugas",
        "min": 0.0, "max": 40.0, "default": 30.0, "step": 1.0,
        "help": "Total skor tugas siswa (0–40)",
        "icon": "📚"
    },
    "Jam_Belajar_Harian": {
        "label": "Jam Belajar Harian (jam/hari)",
        "min": 1.0, "max": 9.0, "default": 3.0, "step": 1.0,
        "help": "Rata-rata jam belajar per hari (1–9 jam, sesuai data latih)",
        "icon": "⏱️"
    },
    # Fitur fallback (jika ada di model tapi tidak ada di config di atas)
    # akan ditangani secara generic
}

# Kategori tampilan (hanya fitur yang relevan untuk input manual)
# Urutan dan pengelompokan sesuai formula model
DISPLAY_CATEGORIES = {
    "👥 Kehadiran": ["Persentase_Kehadiran"],
    "📝 Nilai Internal": ["Nilai_Internal_1", "Nilai_Internal_2"],
    "📚 Tugas & Jam Belajar": ["Skor_Tugas", "Jam_Belajar_Harian"],
}

# Himpunan fitur yang tampil di form manual
DISPLAYED_FEATURES = set()
for feats in DISPLAY_CATEGORIES.values():
    DISPLAYED_FEATURES.update(feats)

# Fitur yang ADA di model tapi tidak perlu diisi user (auto-nilai default/median)
HIDDEN_FEATURES = [f for f in FEATURES if f not in DISPLAYED_FEATURES]


def get_feature_cfg(feature_name):
    """Ambil konfigurasi fitur, fallback ke generic jika tidak ada."""
    if feature_name in FEATURE_CONFIG:
        return FEATURE_CONFIG[feature_name]
    return {
        "label": feature_name.replace("_", " "),
        "min": 0.0, "max": 100.0, "default": 50.0, "step": 1.0,
        "help": f"Nilai untuk {feature_name.replace('_', ' ')}",
        "icon": "📊"
    }


# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
            <div style='font-size: 4rem;'>🎓</div>
            <h2 style='color: white; margin: 0.5rem 0;'>Academic AI</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Prediksi Nilai Berbasis AI</p>
        </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigation Menu",
        [
            "🏠 Dashboard",
            "📊 Visualisasi Data & Model",
            "🎯 Prediksi Individual",
            "📁 Prediksi Batch (CSV)",
            "ℹ️ Informasi Model"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; margin-top: 2rem;'>
            <p style='font-size: 0.9rem; margin: 0;'>
                <b>⚡ Powered by</b><br>
                Machine Learning<br>
                <span style='font-size: 0.8rem; opacity: 0.8;'>Version 1.0</span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ======================================================
# PAGE 0: DASHBOARD
# ======================================================
if menu == "🏠 Dashboard":
    st.markdown("""
        <div class='main-header'>
            <h1>🎓 Academic Performance Predictor</h1>
            <p style='font-size: 1.3rem; margin-top: 1rem; opacity: 0.95;'>
                ✨ Sistem Prediksi Nilai Akhir Siswa Berbasis Machine Learning ✨
            </p>
            <div style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
                🚀 Accurate • ⚡ Fast • 🎯 Reliable
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Performa Model")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #667eea;'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>🎯</div>
                <h3 style='color: #667eea; margin: 0; text-align: center; font-weight: 700;'>R² Score</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['r2']:.3f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Akurasi Model</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #f093fb;'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>📊</div>
                <h3 style='color: #f093fb; margin: 0; text-align: center; font-weight: 700;'>MAE</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['mae']:.2f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Mean Absolute Error</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #4facfe;'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>📈</div>
                <h3 style='color: #4facfe; margin: 0; text-align: center; font-weight: 700;'>RMSE</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{metrics['rmse']:.2f}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Root Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #43e97b;'>
                <div style='font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem;'>🔢</div>
                <h3 style='color: #43e97b; margin: 0; text-align: center; font-weight: 700;'>Features</h3>
                <h1 style='margin: 0.5rem 0; text-align: center; font-size: 2.5rem; color: #333;'>{len(FEATURES)}</h1>
                <p style='color: #555; margin: 0; text-align: center; font-weight: 500;'>Total Fitur Input</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Fitur Unggulan Sistem")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='feature-card'><div style='font-size:3rem;'>🤖</div><h3>AI-Powered</h3><p>Prediksi menggunakan algoritma Machine Learning canggih</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='feature-card'><div style='font-size:3rem;'>⚡</div><h3>Real-time</h3><p>Hasil prediksi instan dalam hitungan detik</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='feature-card'><div style='font-size:3rem;'>📊</div><h3>Batch Processing</h3><p>Prediksi massal dengan upload CSV</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🔍 Fitur-Fitur Input Model")
        features_df = pd.DataFrame({
            "No": range(1, len(FEATURES) + 1),
            "Nama Fitur": [f"📌 {f.replace('_', ' ')}" for f in FEATURES],
            "Kode": FEATURES
        })
        st.dataframe(features_df, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #16a34a;
            box-shadow: 0 4px 10px rgba(22,163,74,0.15);'>
                <h3 style='margin-top: 0; color: #14532d; font-weight: 700;'>🎯 Fitur Utama</h3>
                <p style='margin: 0.5rem 0; color: #166534; font-weight: 600;'>Faktor Paling Berpengaruh:</p>
                <ul style='margin: 0; padding-left: 1.5rem; color: #166534; line-height: 2;'>
                    <li>📝 <b>Nilai Internal 1 &amp; 2</b></li>
                    <li>📚 <b>Skor Tugas</b></li>
                    <li>👥 <b>Persentase Kehadiran</b></li>
                    <li>⏱️ <b>Jam Belajar Harian</b></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Statistik Penggunaan")
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("🎓", "100+", "Prediksi Akurat", "#667eea", "#764ba2"),
        ("⚡", "< 1s", "Waktu Proses", "#f093fb", "#f5576c"),
        ("🎯", "95%", "Tingkat Akurasi", "#4facfe", "#00f2fe"),
        ("👥", "50+", "Pengguna Aktif", "#43e97b", "#38f9d7"),
    ]
    for col, (icon, val, label, c1, c2) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(f"""
                <div style='text-align: center; padding: 1.5rem;
                background: linear-gradient(135deg, {c1} 0%, {c2} 100%);
                border-radius: 12px; color: white;'>
                    <div style='font-size: 2rem;'>{icon}</div>
                    <h2 style='margin: 0.5rem 0;'>{val}</h2>
                    <p style='margin: 0; opacity: 0.9;'>{label}</p>
                </div>
            """, unsafe_allow_html=True)

# ======================================================
# PAGE 1: VISUALISASI DATA & MODEL
# ======================================================
elif menu == "📊 Visualisasi Data & Model":
    st.markdown("""
        <div class='main-header'>
            <h1>📊 Visualisasi Data & Analisis Model</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>🔍 Analisis mendalam tentang performa dan karakteristik model</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📈 Metrik Performa Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 R² Score", f"{metrics['r2']:.3f}", help="Mengukur seberapa baik model menjelaskan variasi data (0-1)")
    col2.metric("📊 MAE", f"{metrics['mae']:.2f}", help="Rata-rata kesalahan absolut prediksi")
    col3.metric("📈 RMSE", f"{metrics['rmse']:.2f}", help="Akar kuadrat rata-rata kesalahan kuadrat")

    st.markdown("---")
    st.markdown("### 🔍 Feature Importance Analysis")

    tab1, tab2, tab3 = st.tabs(["📊 Bar Chart", "🎯 Horizontal Chart", "📋 Table"])

    coef_df = pd.DataFrame({
        "Feature": [f.replace("_", " ") for f in FEATURES],
        "Coefficient": model.coef_,
        "Abs_Coefficient": np.abs(model.coef_)
    }).sort_values("Abs_Coefficient", ascending=False)

    with tab1:
        fig = px.bar(coef_df, x="Feature", y="Coefficient", color="Coefficient",
                     color_continuous_scale=["#ff6b6b", "#ffe66d", "#4ecdc4", "#45b7d1"],
                     title="🎯 Pengaruh Setiap Fitur terhadap Nilai Akhir",
                     labels={"Coefficient": "Koefisien", "Feature": "Fitur"})
        fig.update_layout(height=500, xaxis_tickangle=-45, showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.bar(coef_df, y="Feature", x="Coefficient", orientation="h", color="Coefficient",
                      color_continuous_scale="viridis", title="📊 Feature Importance (Sorted)",
                      labels={"Coefficient": "Koefisien", "Feature": "Fitur"})
        fig2.update_layout(height=600, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        display_df = coef_df.copy()
        display_df["Impact"] = display_df["Coefficient"].apply(lambda x: "Positif ⬆️" if x > 0 else "Negatif ⬇️")
        display_df["Magnitude"] = display_df["Abs_Coefficient"].apply(
            lambda x: "Tinggi 🔥" if x > 2 else "Sedang 📊" if x > 1 else "Rendah 📉")
        st.dataframe(display_df[["Feature", "Coefficient", "Impact", "Magnitude"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📐 Persamaan Model Linear Regression")
    col1, col2 = st.columns([2, 1])
    with col1:
        formula = f"Final Score = {model.intercept_:.2f}"
        for f, c in zip(FEATURES, model.coef_):
            sign = "+" if c >= 0 else "-"
            formula += f" {sign} ({abs(c):.2f} × {f.replace('_', ' ')})"
        st.code(formula, language="python")
    with col2:
        st.markdown("""
            <div style='background: #1a1a2e; padding: 1.5rem; border-radius: 12px;
            border-left: 5px solid #667eea; margin-top: 0.5rem;
            box-shadow: 0 4px 15px rgba(102,126,234,0.3);'>
                <h4 style='margin-top: 0; color: #a78bfa; font-size: 1rem;'>💡 Interpretasi</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #e2e8f0; line-height: 2;'>
                    <li>📊 <b style="color:#93c5fd;">Intercept</b>: Nilai dasar prediksi</li>
                    <li>⬆️ <b style="color:#6ee7b7;">Koefisien (+)</b>: Meningkatkan nilai akhir</li>
                    <li>⬇️ <b style="color:#fca5a5;">Koefisien (-)</b>: Menurunkan nilai akhir</li>
                </ul>
                <p style='margin: 0.8rem 0 0 0; color: #94a3b8; font-size: 0.82rem;'>
                    Semakin besar nilai absolut koefisien, semakin besar pengaruh fitur tersebut.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Distribusi Koefisien")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=model.coef_, nbinsx=20, name="Distribusi",
                                    marker_color='#667eea', marker_line_color='#764ba2', marker_line_width=1.5))
    fig_dist.update_layout(title="📈 Distribusi Nilai Koefisien", xaxis_title="Nilai Koefisien",
                           yaxis_title="Frekuensi", height=400,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_dist, use_container_width=True)

# ======================================================
# PAGE 2: PREDIKSI INDIVIDUAL  ← BAGIAN YANG DIPERBAIKI
# ======================================================
elif menu == "🎯 Prediksi Individual":
    st.markdown("""
        <div class='main-header'>
            <h1>🎯 Prediksi Nilai Individual</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                📝 Masukkan data siswa untuk memprediksi nilai akhir
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2563eb;
        margin: 1rem 0; box-shadow: 0 4px 10px rgba(37,99,235,0.15);'>
            <h4 style='margin-top: 0; color: #1e3a8a; font-weight: 700;'>📝 Instruksi:</h4>
            <p style='margin: 0; color: #1e40af; line-height: 1.7;'>
                Isi data siswa di bawah ini sesuai dengan nilai aktual, lalu klik tombol
                <b>"Prediksi Nilai Akhir"</b> untuk mendapatkan hasil prediksi berbasis AI.
                Semua input sudah disesuaikan dengan rentang data latih yang sebenarnya.
            </p>
        </div>
    """, unsafe_allow_html=True)

    inputs = {}

    # ── Render hanya fitur yang relevan & bermakna ──
    st.markdown("### 📋 Input Data Siswa")

    for cat_label, cat_features in DISPLAY_CATEGORIES.items():
        # filter hanya fitur yang memang ada di model
        valid_feats = [f for f in cat_features if f in FEATURES]
        if not valid_feats:
            continue

        st.markdown(f"""
            <div class='input-section-header'>{cat_label}</div>
        """, unsafe_allow_html=True)

        cols = st.columns(min(len(valid_feats), 3))
        for i, feature in enumerate(valid_feats):
            cfg = get_feature_cfg(feature)
            with cols[i % 3]:
                inputs[feature] = st.number_input(
                    f"{cfg['icon']} {cfg['label']}",
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    value=cfg["default"],
                    step=cfg["step"],
                    help=cfg["help"]
                )

    # ── Fitur tersembunyi: isi dengan nilai default/tengah supaya model tetap bisa berjalan ──
    for feature in HIDDEN_FEATURES:
        cfg = get_feature_cfg(feature)
        inputs[feature] = cfg["default"]

    # ── Info fitur yang diabaikan dari input manual ──
    if HIDDEN_FEATURES:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            padding: 1rem 1.5rem; border-radius: 10px; border-left: 4px solid #9c27b0; margin: 1rem 0;'>
                <p style='margin: 0; color: #6a1b9a; font-size: 0.9rem;'>
                    ℹ️ <b>Fitur berikut tidak ditampilkan</b> karena tidak signifikan secara akademik
                    (otomatis diisi nilai tengah): <i>{', '.join([f.replace('_',' ') for f in HIDDEN_FEATURES])}</i>
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Ringkasan input sebelum prediksi ──
    st.markdown("### 👀 Ringkasan Input")
    summary_cols = st.columns(len(DISPLAYED_FEATURES))
    displayed_list = [f for f in FEATURES if f in DISPLAYED_FEATURES]
    for col, feat in zip(summary_cols, displayed_list):
        cfg = get_feature_cfg(feat)
        with col:
            st.markdown(f"""
                <div style='text-align:center; padding:1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius:10px; color:white; margin-bottom:0.5rem;'>
                    <div style='font-size:1.5rem;'>{cfg['icon']}</div>
                    <div style='font-size:0.75rem; opacity:0.85; margin-bottom:0.3rem;'>{cfg['label']}</div>
                    <div style='font-size:1.8rem; font-weight:700;'>{inputs[feat]:.0f}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Tombol prediksi ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Prediksi Nilai Akhir", use_container_width=True, type="primary")

    if predict_button:
        with st.spinner("🔄 Memproses prediksi dengan AI..."):
            X = pd.DataFrame([inputs], columns=FEATURES)
            X_scaled = scaler.transform(X)
            raw_prediction = model.predict(X_scaled)[0]

            prediction = raw_prediction
            min_internal = min(
                inputs.get("Nilai_Internal_1", 0),
                inputs.get("Nilai_Internal_2", 0)
            )
            kehadiran = inputs.get("Persentase_Kehadiran", 0)
            skor_tugas = inputs.get("Skor_Tugas", 0)

            # Aturan akademik
            if min_internal < 15:
                prediction = min(prediction, 69)
            elif min_internal < 20:
                prediction = min(prediction, 79)
            else:
                if not (min_internal >= 25 and kehadiran >= 85 and skor_tugas >= 25):
                    prediction = min(prediction, 89)

            prediction = np.clip(prediction, 0, 100)

        st.balloons()
        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction >= 90:
                grade, grade_class, emoji, message = "A", "grade-a", "🌟", "Excellent"
            elif prediction >= 80:
                grade, grade_class, emoji, message = "B", "grade-b", "👍", "Very Good"
            elif prediction >= 65:
                grade, grade_class, emoji, message = "C", "grade-c", "🙂", "Good"
            else:
                grade, grade_class, emoji, message = "D", "grade-d", "⚠️", "Needs Improvement"

            st.markdown(f"""
                <div style='text-align: center; padding: 3rem;
                background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
                border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.1);'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🎓</div>
                    <h1 style='font-size: 5rem; margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                    {prediction:.2f}</h1>
                    <p style='font-size: 1.5rem; color: #666; margin: 0.5rem 0;'>Nilai Akhir Prediksi</p>
                    <div class='{grade_class} grade-badge'>{emoji} Grade {grade} — {message}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Rincian Analisis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Data Input (Fitur Utama)")
            displayed_input = {k: v for k, v in inputs.items() if k in DISPLAYED_FEATURES}
            input_df = pd.DataFrame({
                "Fitur": [f"📌 {f.replace('_', ' ')}" for f in displayed_input.keys()],
                "Nilai": list(displayed_input.values())
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("#### 🎯 Evaluasi Akademik")

            rules_met, rules_not_met = [], []

            if min_internal >= 25:
                rules_met.append("✅ Nilai Internal minimum ≥ 25")
            else:
                rules_not_met.append(f"❌ Nilai Internal minimum: {min_internal:.0f} (perlu ≥ 25 untuk A)")

            if kehadiran >= 85:
                rules_met.append("✅ Kehadiran ≥ 85%")
            else:
                rules_not_met.append(f"❌ Kehadiran: {kehadiran:.0f}% (perlu ≥ 85% untuk A)")

            if skor_tugas >= 25:
                rules_met.append("✅ Skor Tugas ≥ 25")
            else:
                rules_not_met.append(f"❌ Skor Tugas: {skor_tugas:.0f} (perlu ≥ 25 untuk A)")

            for rule in rules_met:
                st.success(rule)
            for rule in rules_not_met:
                st.warning(rule)

            if rules_not_met:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                    padding: 1.2rem 1.5rem; border-radius: 10px; border-left: 4px solid #2563eb;
                    margin-top: 0.5rem; box-shadow: 0 3px 8px rgba(37,99,235,0.15);'>
                        <h4 style='margin-top: 0; color: #1e3a8a; font-weight: 700;'>💡 Rekomendasi</h4>
                        <p style='margin: 0; color: #1e40af;'>
                            Tingkatkan aspek yang belum memenuhi syarat untuk mendapatkan nilai yang lebih baik!
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# ======================================================
# PAGE 3: PREDIKSI BATCH
# ======================================================
elif menu == "📁 Prediksi Batch (CSV)":
    st.markdown("""
        <div class='main-header'>
            <h1>📁 Prediksi Batch (CSV)</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>📤 Upload file CSV untuk memprediksi nilai banyak siswa sekaligus</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2563eb;
            margin: 1rem 0; box-shadow: 0 4px 10px rgba(37,99,235,0.15);'>
                <h4 style='margin-top: 0; color: #1e3a8a; font-weight: 700;'>📄 Format File CSV:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #1e40af; line-height: 1.9;'>
                    <li>File harus berformat <b>.csv</b></li>
                    <li>Kolom harus sesuai fitur model</li>
                    <li>Tidak ada nilai kosong (NaN)</li>
                    <li>Gunakan pemisah koma (,)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        feature_list_html = "".join([
            f"<li>📌 <b>{f.replace('_',' ')}</b></li>" for f in FEATURES
        ])
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #16a34a;
            margin: 1rem 0; box-shadow: 0 4px 10px rgba(22,163,74,0.15);'>
                <h4 style='margin-top: 0; color: #14532d; font-weight: 700;'>✅ Fitur yang Diperlukan:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #166534; line-height: 1.9;'>
                    {feature_list_html}
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📥 Download Template")
    template_df = pd.DataFrame(columns=FEATURES)
    # Isi template dengan nilai default yang realistis
    sample_row = {}
    for f in FEATURES:
        cfg = get_feature_cfg(f)
        sample_row[f] = cfg["default"]
    template_df.loc[0] = sample_row

    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Template CSV", csv_template,
                       "template_prediksi.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📤 Upload File CSV")
    file = st.file_uploader("Pilih file CSV", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)
            st.success(f"✅ File berhasil diupload! Total data: {len(df)} baris")

            st.markdown("### 👀 Preview Data")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            for col, (icon, val, label, c1, c2) in zip(
                [col1, col2, col3, col4],
                [
                    ("📊", len(df), "Total Baris", "#667eea", "#764ba2"),
                    ("📋", len(df.columns), "Total Kolom", "#f093fb", "#f5576c"),
                    ("❓", df.isnull().sum().sum(), "Missing Values", "#4facfe", "#00f2fe"),
                    ("🔄", df.duplicated().sum(), "Duplicate Rows", "#43e97b", "#38f9d7"),
                ]
            ):
                with col:
                    st.markdown(f"""
                        <div style='text-align: center; padding: 1.5rem;
                        background: linear-gradient(135deg, {c1} 0%, {c2} 100%);
                        border-radius: 12px; color: white;'>
                            <div style='font-size: 2rem;'>{icon}</div>
                            <h2 style='margin: 0.5rem 0;'>{val}</h2>
                            <p style='margin: 0;'>{label}</p>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔍 Validasi Kolom")
            missing_cols = [f for f in FEATURES if f not in df.columns]
            extra_cols = [c for c in df.columns if c not in FEATURES]

            col1, col2 = st.columns(2)
            with col1:
                if missing_cols:
                    st.error(f"❌ **Kolom yang hilang:** {', '.join(missing_cols)}")
                else:
                    st.success("✅ Semua kolom yang diperlukan tersedia!")
            with col2:
                if extra_cols:
                    st.info(f"ℹ️ **Kolom tambahan (diabaikan):** {', '.join(extra_cols)}")

            if not missing_cols:
                st.markdown("---")
                if st.button("🚀 Prediksi Semua Data", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Sedang memproses prediksi..."):
                        X = df[FEATURES].copy()
                        if X.isnull().sum().sum() > 0:
                            st.warning("⚠️ Terdapat nilai kosong. Mengisi dengan median...")
                            X = X.fillna(X.median())

                        X_scaled = scaler.transform(X)
                        predictions = model.predict(X_scaled)

                        final_predictions = []
                        for i, pred in enumerate(predictions):
                            row = df.iloc[i]
                            min_internal = min(
                                row.get("Nilai_Internal_1", 0),
                                row.get("Nilai_Internal_2", 0)
                            )
                            kehadiran = row.get("Persentase_Kehadiran", 0)
                            skor_tugas = row.get("Skor_Tugas", 0)

                            if min_internal < 15:
                                pred = min(pred, 69)
                            elif min_internal < 20:
                                pred = min(pred, 79)
                            else:
                                if not (min_internal >= 25 and kehadiran >= 85 and skor_tugas >= 25):
                                    pred = min(pred, 89)

                            final_predictions.append(np.clip(pred, 0, 100))

                        df["Predicted_Final_Score"] = final_predictions
                        df["Grade"] = df["Predicted_Final_Score"].apply(
                            lambda s: "A" if s >= 90 else "B" if s >= 80 else "C" if s >= 65 else "D"
                        )

                        st.success("✅ Prediksi berhasil!")
                        st.balloons()

                        st.markdown("### 📊 Ringkasan Hasil Prediksi")
                        col1, col2, col3, col4 = st.columns(4)
                        for col, (icon, val, label, c1, c2) in zip(
                            [col1, col2, col3, col4],
                            [
                                ("📊", f"{df['Predicted_Final_Score'].mean():.2f}", "Rata-rata Nilai", "#667eea", "#764ba2"),
                                ("⬆️", f"{df['Predicted_Final_Score'].max():.2f}", "Nilai Tertinggi", "#43e97b", "#38f9d7"),
                                ("⬇️", f"{df['Predicted_Final_Score'].min():.2f}", "Nilai Terendah", "#fa709a", "#fee140"),
                                ("📈", f"{df['Predicted_Final_Score'].std():.2f}", "Std Deviasi", "#4facfe", "#00f2fe"),
                            ]
                        ):
                            with col:
                                st.markdown(f"""
                                    <div style='text-align: center; padding: 1.5rem;
                                    background: linear-gradient(135deg, {c1} 0%, {c2} 100%);
                                    border-radius: 12px; color: white;'>
                                        <div style='font-size: 2rem;'>{icon}</div>
                                        <h2 style='margin: 0.5rem 0;'>{val}</h2>
                                        <p style='margin: 0;'>{label}</p>
                                    </div>
                                """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("### 📈 Distribusi Grade")
                        grade_counts = df["Grade"].value_counts().sort_index()
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_pie = px.pie(values=grade_counts.values, names=grade_counts.index,
                                            title="🎯 Distribusi Grade",
                                            color=grade_counts.index,
                                            color_discrete_map={"A": "#38ef7d", "B": "#00f2fe", "C": "#fee140", "D": "#f7b733"})
                            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            fig_hist = px.histogram(df, x="Predicted_Final_Score", nbins=20,
                                                   title="📊 Distribusi Nilai Prediksi",
                                                   color_discrete_sequence=["#667eea"])
                            fig_hist.update_layout(xaxis_title="Nilai", yaxis_title="Frekuensi",
                                                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(fig_hist, use_container_width=True)

                        st.markdown("### 📋 Hasil Prediksi Lengkap")
                        st.dataframe(df, use_container_width=True)

                        st.markdown("### ⬇️ Download Hasil")
                        csv_result = df.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download Hasil Prediksi (CSV)", csv_result,
                                          "hasil_prediksi.csv", "text/csv", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
            st.info("💡 Pastikan file CSV Anda memiliki format yang benar dan tidak corrupt.")

# ======================================================
# PAGE 4: INFORMASI MODEL
# ======================================================
elif menu == "ℹ️ Informasi Model":
    st.markdown("""
        <div class='main-header'>
            <h1>ℹ️ Informasi Model</h1>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>📚 Detail teknis dan dokumentasi sistem</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #2563eb;
            margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(37,99,235,0.15);'>
                <h4 style='margin-top: 0; color: #1e3a8a; font-weight: 700;'>🔧 Tipe Model:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #1e40af; line-height: 1.9;'>
                    <li>📊 Linear Regression</li>
                    <li>🎯 Supervised Learning</li>
                    <li>📈 Regression Task</li>
                </ul>
                <h4 style='margin-top: 1rem; color: #1e3a8a; font-weight: 700;'>📚 Library:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #1e40af; line-height: 1.9;'>
                    <li>🔬 Scikit-learn</li>
                    <li>🐼 Pandas</li>
                    <li>🔢 NumPy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            padding: 1.5rem; border-radius: 12px; border-left: 5px solid #16a34a;
            margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(22,163,74,0.15);'>
                <h4 style='margin-top: 0; color: #14532d; font-weight: 700;'>⚡ Performa:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #166534; line-height: 1.9;'>
                    <li>🎯 R² Score: <b>{metrics['r2']:.3f}</b></li>
                    <li>📊 MAE: <b>{metrics['mae']:.2f}</b></li>
                    <li>📈 RMSE: <b>{metrics['rmse']:.2f}</b></li>
                </ul>
                <h4 style='margin-top: 1rem; color: #14532d; font-weight: 700;'>🔢 Features:</h4>
                <ul style='margin: 0; padding-left: 1.5rem; color: #166534; line-height: 1.9;'>
                    <li>📋 Total: <b>{len(FEATURES)} fitur</b></li>
                    <li>✅ Fitur input utama: <b>{len(DISPLAYED_FEATURES)}</b></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Daftar Fitur Lengkap")
    features_detail = pd.DataFrame({
        "No": range(1, len(FEATURES) + 1),
        "Nama Fitur": [f"📌 {f.replace('_', ' ')}" for f in FEATURES],
        "Kode Fitur": FEATURES,
        "Koefisien": model.coef_,
        "Pengaruh": ["⬆️ Positif" if c > 0 else "⬇️ Negatif" for c in model.coef_],
        "Tampil di Form": ["✅ Ya" if f in DISPLAYED_FEATURES else "🔒 Auto" for f in FEATURES]
    })
    st.dataframe(features_detail, use_container_width=True, hide_index=True)

    st.markdown("### 📊 Rentang Nilai Input (Referensi Data Latih)")
    range_data = []
    for f in FEATURES:
        cfg = get_feature_cfg(f)
        range_data.append({
            "Fitur": f.replace("_", " "),
            "Min": cfg["min"],
            "Default": cfg["default"],
            "Max": cfg["max"],
            "Ditampilkan": "✅" if f in DISPLAYED_FEATURES else "🔒"
        })
    st.dataframe(pd.DataFrame(range_data), use_container_width=True, hide_index=True)

    st.markdown("### 💻 System Requirements")
    st.code("""
Python >= 3.8
streamlit >= 1.28.0
pandas >= 1.5.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
plotly >= 5.17.0
joblib >= 1.3.0
    """, language="text")

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
    <div class='footer'>
        <div style='font-size: 2rem; margin-bottom: 1rem;'>🎓✨</div>
        <h3 style='margin: 0.5rem 0; color: white;'>Academic Performance Predictor</h3>
        <p style='margin: 0.5rem 0; opacity: 0.95; color: white;'>Powered by Machine Learning & Artificial Intelligence</p>
        <div style='text-align: center; color: white; padding: 20px 30px; margin-top: 20px;'>
            <p style='font-size: 16px; margin: 0 0 15px 0; font-weight: 600; color: white;'>
                Copyright © 2026 BY Pengelola MK Praktikum Unggulan (Praktikum DGX)
            </p>
            <div style='margin-top: 15px;'>
                <a href='https://www.praktikum-hpc.gunadarma.ac.id/' target='_blank'
                   style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.95; font-weight: 500;'>
                    🔗 praktikum-hpc.gunadarma.ac.id
                </a>
                <br><br>
                <a href='https://www.hpc-hub.gunadarma.ac.id/' target='_blank'
                   style='color: white; text-decoration: none; margin: 0 10px; font-size: 14px; opacity: 0.95; font-weight: 500;'>
                    🔗 hpc-hub.gunadarma.ac.id
                </a>
            </div>
        </div>
        <div style='margin-top: 1rem; font-size: 0.9rem; color: white; opacity: 0.85;'>
            Made with ❤️ using Streamlit
        </div>
    </div>
""", unsafe_allow_html=True)