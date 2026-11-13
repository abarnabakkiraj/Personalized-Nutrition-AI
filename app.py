"""
app.py - Personalized Nutrition AI from Blood Test Reports
Main Streamlit application entry point.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
import io, os, json
from datetime import datetime

# ── Local module imports ──────────────────────────────────────────────────────
from database import (
    init_db, register_user, login_user,
    save_blood_report, get_user_reports,
    save_recommendations, get_recommendations,
    save_prediction, get_prediction_history,
)
from ocr_reader import parse_uploaded_file, generate_sample_csv
from nutrition_engine import (
    analyze_blood_report, calculate_health_score,
    get_risk_level, NORMAL_RANGES, PARAMETER_LABELS,
)
from ml_model import predict_risk, get_feature_importance
from report_generator import generate_pdf_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nutrition AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize DB ─────────────────────────────────────────────────────────────
init_db()

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "user": None,
    "page": "Login",
    "analysis_results": None,
    "current_report_id": None,
    "blood_values": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a5276 0%, #148f77 100%);
        color: white;
    }
    section[data-testid="stSidebar"] .stRadio label { color: white !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: white !important; }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
        border-left: 5px solid #148f77;
    }
    .card-low    { border-left-color: #e74c3c !important; }
    .card-high   { border-left-color: #e67e22 !important; }
    .card-normal { border-left-color: #27ae60 !important; }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: white;
    }
    .badge-Low    { background: #e74c3c; }
    .badge-High   { background: #e67e22; }
    .badge-Normal { background: #27ae60; }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1a5276;
        border-bottom: 2px solid #148f77;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <h1 style='color:#1a5276; font-size:36px;'>🩺 Nutrition AI</h1>
            <p style='color:#7f8c8d; font-size:16px;'>AI-Powered Blood Report Analysis & Nutrition Planning</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Login", use_container_width=False, type="primary")

            if submitted:
                user = login_user(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user      = user
                    st.session_state.page      = "Dashboard"
                    st.success(f"Welcome back, {user.get('full_name', username)}! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tab_register:
            with st.form("register_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_user  = st.text_input("Username*")
                    full_name = st.text_input("Full Name*")
                    age       = st.number_input("Age", min_value=1, max_value=120, value=25)
                with col_b:
                    new_pass  = st.text_input("Password*", type="password")
                    conf_pass = st.text_input("Confirm Password*", type="password")
                    gender    = st.selectbox("Gender", ["Male", "Female", "Other"])
                email     = st.text_input("Email (optional)")
                reg_btn   = st.form_submit_button("Create Account", use_container_width=False, type="primary")

            if reg_btn:
                if not new_user or not new_pass:
                    st.error("Username and Password are required.")
                elif new_pass != conf_pass:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    ok, msg = register_user(new_user, new_pass, full_name, age, gender, email)
                    if ok:
                        st.success(msg + " Please log in.")
                    else:
                        st.error(msg)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        user = st.session_state.user
        st.markdown(f"""
        <div style='text-align:center; padding:10px;'>
            <div style='font-size:48px;'>👤</div>
            <h3 style='color:white; margin:5px 0;'>{user.get('full_name', user['username'])}</h3>
            <p style='color:#aed6f1; font-size:13px;'>
                {user.get('age', '')} yrs • {user.get('gender', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        pages = {
            "📊 Dashboard":        "Dashboard",
            "📤 Upload Report":    "Upload",
            "✏️  Manual Entry":    "Manual",
            "🔬 Analysis":         "Analysis",
            "🤖 ML Prediction":    "Prediction",
            "📈 History":          "History",
            "📄 Download Report":  "Download",
        }

        for label, page in pages.items():
            if st.button(label, use_container_width=False,
                         type="primary" if st.session_state.page == page else "secondary"):
                st.session_state.page = page
                st.rerun()

        st.divider()
        if st.button("🚪 Logout", use_container_width=False):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown('<div class="section-header">📊 Dashboard</div>', unsafe_allow_html=True)
    user = st.session_state.user

    # Quick stats
    reports = get_user_reports(user["id"])
    preds   = get_prediction_history(user["id"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reports",   len(reports))
    c2.metric("Analyses Done",   len(preds))

    if preds:
        latest = preds[0]
        c3.metric("Latest Health Score", f"{latest['health_score']}/100")
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(latest["risk_level"], "⚪")
        c4.metric("Risk Level", f"{risk_emoji} {latest['risk_level']}")
    else:
        c3.metric("Latest Health Score", "N/A")
        c4.metric("Risk Level", "N/A")

    st.divider()

    # Last analysis quick view
    if st.session_state.analysis_results:
        st.subheader("Last Analysis Overview")
        results = st.session_state.analysis_results
        _render_parameter_cards(results)

        # Quick bar chart
        _render_bar_chart(results)
    else:
        st.info("No analysis yet. Upload a blood report or enter values manually to get started.")
        if st.button("➕ Enter Blood Values", type="primary"):
            st.session_state.page = "Manual"
            st.rerun()

    # Recent prediction history table
    if preds:
        st.subheader("Recent Predictions")
        df = pd.DataFrame(preds)[["predicted_at", "risk_level", "health_score", "model_version"]]
        df.columns = ["Date", "Risk Level", "Health Score", "Model"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y %H:%M")
        st.dataframe(df.head(5), use_container_width=False)


def _render_parameter_cards(results):
    cols = st.columns(4)
    for i, row in enumerate(results):
        css_class = f"card-{row['status'].lower()}"
        badge_cls  = f"badge-{row['status']}"
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <small style='color:#7f8c8d;'>{row['label']}</small><br>
                <span style='font-size:22px; font-weight:700;'>{row['value']}</span>
                <span style='font-size:12px; color:#95a5a6;'> {row['unit']}</span><br>
                <span class="badge {badge_cls}">{row['status']}</span>
                <br><small style='color:#aaa;'>Normal: {row['normal_range']}</small>
            </div>
            """, unsafe_allow_html=True)


def _render_bar_chart(results):
    labels  = [r["label"] for r in results]
    values  = [r["value"] for r in results]
    colors  = {
        "Low": "#e74c3c", "High": "#e67e22", "Normal": "#27ae60"
    }
    bar_colors = [colors[r["status"]] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Blood Parameter Values", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=30)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    # Legend
    patches = [
        mpatches.Patch(color="#e74c3c", label="Low"),
        mpatches.Patch(color="#27ae60", label="Normal"),
        mpatches.Patch(color="#e67e22", label="High"),
    ]
    ax.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_upload():
    st.markdown('<div class="section-header">📤 Upload Blood Report</div>', unsafe_allow_html=True)

    st.info("Upload a blood test report in PDF, image (JPG/PNG), or CSV format. The AI will extract blood parameter values automatically.")

    # Sample CSV download
    with st.expander("📋 Download CSV Template"):
        sample_csv = generate_sample_csv()
        st.download_button(
            "⬇️ Download Sample CSV",
            data=sample_csv,
            file_name="blood_report_template.csv",
            mime="text/csv",
        )
        st.code(sample_csv, language="csv")

    uploaded = st.file_uploader(
        "Choose your blood report file",
        type=["pdf", "jpg", "jpeg", "png", "csv", "bmp"],
        help="PDF, Image (JPG/PNG/BMP), or CSV format supported."
    )

    if uploaded:
        file_bytes = uploaded.read()
        with st.spinner("🔍 Extracting blood values using OCR/parsing..."):
            extracted, raw_text = parse_uploaded_file(file_bytes, uploaded.name)

        if extracted:
            st.success(f"✅ Successfully extracted {len(extracted)} blood parameters!")
            st.session_state.blood_values = extracted

            # Preview extracted values
            st.subheader("Extracted Values")
            df_preview = pd.DataFrame([
                {"Parameter": PARAMETER_LABELS.get(k, k), "Value": v}
                for k, v in extracted.items()
            ])
            st.dataframe(df_preview, use_container_width=False)

            # Allow manual correction
            st.subheader("Review & Correct (optional)")
            corrected = {}
            cols = st.columns(2)
            for i, (param, val) in enumerate(extracted.items()):
                lo, hi, unit = NORMAL_RANGES.get(param, (0, 999, ""))
                with cols[i % 2]:
                    corrected[param] = st.number_input(
                        f"{PARAMETER_LABELS.get(param, param)} ({unit})",
                        value=float(val),
                        key=f"upload_{param}",
                        step=0.1,
                    )

            if st.button("🔬 Analyse Report", type="primary", use_container_width=False):
                _run_analysis(corrected, source_type="ocr")

        else:
            st.warning("⚠️ Could not automatically extract values. Please use Manual Entry instead.")
            if isinstance(raw_text, str) and raw_text:
                with st.expander("Raw extracted text"):
                    st.text(raw_text[:2000])


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL ENTRY PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_manual():
    st.markdown('<div class="section-header">✏️ Manual Blood Value Entry</div>', unsafe_allow_html=True)
    st.info("Enter blood parameter values from your report. Leave any unknown values at 0 to skip them.")

    with st.form("manual_form"):
        c1, c2 = st.columns(2)

        with c1:
            hemoglobin    = st.number_input("Hemoglobin (g/dL)",         0.0, 30.0,  13.5, 0.1)
            glucose       = st.number_input("Blood Glucose (mg/dL)",      0.0, 600.0, 95.0, 1.0)
            cholesterol   = st.number_input("Total Cholesterol (mg/dL)",  0.0, 500.0, 185.0,1.0)
            vitamin_d     = st.number_input("Vitamin D (ng/mL)",          0.0, 200.0, 28.0, 0.5)

        with c2:
            iron          = st.number_input("Serum Iron (µg/dL)",         0.0, 400.0, 100.0,1.0)
            calcium       = st.number_input("Calcium (mg/dL)",            0.0, 20.0,  9.5,  0.1)
            vitamin_b12   = st.number_input("Vitamin B12 (pg/mL)",        0.0, 2000.0,450.0,10.0)
            triglycerides = st.number_input("Triglycerides (mg/dL)",      0.0, 1000.0,140.0,1.0)

        submitted = st.form_submit_button("🔬 Analyse", type="primary", use_container_width=False)

    if submitted:
        values = {
            "hemoglobin":    hemoglobin,
            "glucose":       glucose,
            "cholesterol":   cholesterol,
            "vitamin_d":     vitamin_d,
            "iron":          iron,
            "calcium":       calcium,
            "vitamin_b12":   vitamin_b12,
            "triglycerides": triglycerides,
        }
        # Remove zero-value entries (user skipped)
        values = {k: v for k, v in values.items() if v > 0}
        _run_analysis(values, source_type="manual")


def _run_analysis(values: dict, source_type: str = "manual"):
    """Shared analysis runner called from Upload and Manual pages."""
    user = st.session_state.user

    with st.spinner("🤖 Running AI analysis..."):
        # Save to DB
        report_id = save_blood_report(user["id"], values, source_type)

        # Analyse
        results = analyze_blood_report(values)
        score   = calculate_health_score(results)
        risk    = get_risk_level(score)

        # Save recommendations
        recs_to_save = [{"parameter": r["parameter"], "status": r["status"],
                         "recommendation": r["foods"]} for r in results]
        save_recommendations(user["id"], report_id, recs_to_save)

        # ML prediction
        pred = predict_risk(values)
        save_prediction(user["id"], report_id, pred["risk_level"], score)

        # Store in session
        st.session_state.analysis_results   = results
        st.session_state.current_report_id  = report_id
        st.session_state.blood_values       = values

    st.success("Analysis complete! Navigate to **🔬 Analysis** to view your results.")
    st.session_state.page = "Analysis"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_analysis():
    st.markdown('<div class="section-header">🔬 Blood Report Analysis</div>', unsafe_allow_html=True)

    results = st.session_state.analysis_results
    if not results:
        st.warning("No analysis data. Please upload a report or enter values manually first.")
        return

    score = calculate_health_score(results)
    risk  = get_risk_level(score)

    # ── Health Score Gauge ────────────────────────────────────────────────────
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Health Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#148f77"},
            "steps": [
                {"range": [0,  50],  "color": "#fadbd8"},
                {"range": [50, 80],  "color": "#fdebd0"},
                {"range": [80, 100], "color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "#1a5276", "width": 3}, "value": score},
        },
        number={"suffix": "/100"},
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=False)

    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
    st.markdown(f"### Risk Level: {risk_emoji} **{risk}**")

    st.divider()

    # ── Parameter Cards ───────────────────────────────────────────────────────
    st.subheader("Parameter Status")
    _render_parameter_cards(results)

    st.divider()

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("Parameter Radar")
    _render_radar_chart(results)

    st.divider()

    # ── Detailed Recommendations ──────────────────────────────────────────────
    st.subheader("📋 Personalized Nutrition Recommendations")

    normal  = [r for r in results if r["status"] == "Normal"]
    abnormal = [r for r in results if r["status"] != "Normal"]

    if normal:
        with st.expander(f"✅ Normal Parameters ({len(normal)})"):
            for r in normal:
                st.success(f"**{r['label']}**: {r['value']} {r['unit']} — Within normal range")

    for rec in abnormal:
        badge = {"Low": "🔴", "High": "🟠"}.get(rec["status"], "⚪")
        with st.expander(f"{badge} {rec['label']} — {rec['status']} ({rec['value']} {rec['unit']})"):
            if rec["foods"]:
                st.markdown("**🥗 Recommended Foods:**")
                cols = st.columns(2)
                for i, food in enumerate(rec["foods"]):
                    cols[i % 2].write(f"• {food}")

            if rec["avoid"]:
                st.markdown("**🚫 Foods to Avoid:**")
                for f in rec["avoid"]:
                    st.write(f"• {f}")

            if rec["tips"]:
                st.markdown("**💡 Health Tips:**")
                for tip in rec["tips"]:
                    st.info(tip)


def _render_radar_chart(results):
    """Plots a radar/spider chart of parameter values normalized to 0-1 scale."""
    labels  = []
    values_norm = []

    for r in results:
        lo, hi, _ = NORMAL_RANGES.get(r["parameter"], (0, 1, ""))
        if hi == lo:
            continue
        # Normalize: 0.5 = normal, <0.5 = low, >0.5 = high
        norm = (r["value"] - lo) / (hi - lo)
        norm = max(0, min(1, norm))
        labels.append(r["label"])
        values_norm.append(norm)

    if not labels:
        return

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(20, 143, 119, 0.2)",
        line=dict(color="#148f77", width=2),
        name="Your Values",
    ))
    # Normal range reference (all 0.5)
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * (len(labels) + 1),
        theta=labels + [labels[0]],
        line=dict(color="#3498db", width=1, dash="dash"),
        name="Normal Midpoint",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
# ML PREDICTION PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_prediction():
    st.markdown('<div class="section-header">🤖 ML Risk Prediction</div>', unsafe_allow_html=True)

    values = st.session_state.blood_values
    if not values:
        st.warning("No blood values found. Please upload or enter values first.")
        return

    with st.spinner("Running ML model..."):
        pred = predict_risk(values)

    risk   = pred["risk_level"]
    proba  = pred["probabilities"]
    conf   = pred["confidence"]

    # Risk display
    colors = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}
    color  = colors.get(risk, "#7f8c8d")
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

    st.markdown(f"""
    <div style='background:{color}20; border:2px solid {color}; border-radius:12px; padding:20px; text-align:center;'>
        <h2 style='color:{color}; margin:0;'>{risk_emoji} {risk} Risk</h2>
        <p style='color:#555; margin:6px 0 0;'>Confidence: {conf*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Probability bar chart
    st.subheader("Risk Probability Breakdown")
    prob_df = pd.DataFrame({
        "Risk Level": list(proba.keys()),
        "Probability": [v * 100 for v in proba.values()],
    })
    fig = px.bar(
        prob_df, x="Risk Level", y="Probability",
        color="Risk Level",
        color_discrete_map={"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"},
        text="Probability",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(yaxis_title="Probability (%)", yaxis_range=[0, 110], showlegend=False)
    st.plotly_chart(fig, use_container_width=False)

    # Feature importance
    st.subheader("Feature Importance (Model Explainability)")
    try:
        fi_df = get_feature_importance()
        fig2 = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Teal",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig2, use_container_width=False)
    except Exception as e:
        st.info(f"Feature importance unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_history():
    st.markdown('<div class="section-header">📈 Report History</div>', unsafe_allow_html=True)
    user = st.session_state.user

    reports = get_user_reports(user["id"])
    preds   = get_prediction_history(user["id"])

    if not reports:
        st.info("No reports yet. Upload or enter blood values to get started.")
        return

    # Health score trend
    if len(preds) > 1:
        st.subheader("Health Score Trend")
        trend_df = pd.DataFrame(preds)
        trend_df["predicted_at"] = pd.to_datetime(trend_df["predicted_at"])
        fig = px.line(
            trend_df.sort_values("predicted_at"),
            x="predicted_at", y="health_score",
            markers=True,
            color_discrete_sequence=["#148f77"],
            labels={"predicted_at": "Date", "health_score": "Health Score"},
        )
        fig.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=False)

    # Reports table
    st.subheader("All Blood Reports")
    display_cols = ["report_date", "hemoglobin", "glucose", "cholesterol",
                    "vitamin_d", "iron", "calcium", "vitamin_b12", "triglycerides"]
    df = pd.DataFrame(reports)[display_cols]
    df.columns = ["Date", "Hb", "Glucose", "Chol", "Vit D", "Iron", "Ca", "B12", "TG"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%d %b %Y %H:%M")
    st.dataframe(df, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD REPORT PAGE
# ══════════════════════════════════════════════════════════════════════════════

def page_download():
    st.markdown('<div class="section-header">📄 Download PDF Report</div>', unsafe_allow_html=True)

    results = st.session_state.analysis_results
    user    = st.session_state.user

    if not results:
        st.warning("No analysis data. Please run an analysis first.")
        return

    score = calculate_health_score(results)
    risk  = get_risk_level(score)

    st.info("Your personalised PDF report will include blood values, deficiency status, nutrition recommendations, and health tips.")

    if st.button("📥 Generate & Download PDF Report", type="primary", use_container_width=False):
        with st.spinner("Generating PDF..."):
            try:
                path = generate_pdf_report(
                    user_info=user,
                    analysis_results=results,
                    health_score=score,
                    risk_level=risk,
                )
                with open(path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="⬇️ Download Report PDF",
                    data=pdf_bytes,
                    file_name=f"nutrition_report_{user['username']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
                st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.info("Tip: Make sure fpdf2 is installed: `pip install fpdf2`")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not st.session_state.logged_in:
        page_login()
        return

    render_sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Upload":
        page_upload()
    elif page == "Manual":
        page_manual()
    elif page == "Analysis":
        page_analysis()
    elif page == "Prediction":
        page_prediction()
    elif page == "History":
        page_history()
    elif page == "Download":
        page_download()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
