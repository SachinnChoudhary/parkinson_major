"""
Model Lab — Machine Learning Training, Validation, and Experiment Registry.

Features:
- CSV dataset upload and column mapping
- Automated dataset-quality and label consistency checks
- Subject-aware train/test splitting with zero-leakage verification
- Comparative training of Logistic Regression, SVM, Random Forest, Extra Trees, Gradient Boosting, XGBoost
- Comprehensive metrics evaluation (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC)
- Model artifact persistence and database registry
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import APP_NAME, DATA_DIR, VOICE_FEATURES, TARGET_COLUMN, SUBJECT_ID_COLUMN
from database.engine import init_db, get_session
from database.models import ModelRegistry, ModelStatus
from ml.dataset import (
    load_dataset, inspect_dataset, extract_subject_id, subject_aware_split, preprocess,
)
from ml.train import (
    get_model_configs, train_single_model, select_best_model, save_model_artifacts,
)
from ml.evaluate import evaluate_model, generate_evaluation_report
from components.charts import model_comparison_chart, confusion_matrix_chart

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"Model Lab — {APP_NAME}",
    layout="wide",
)

init_db()

st.markdown("## Model Engineering & Validation Suite")
st.markdown("""
This laboratory provides end-to-end scientific model development:
upload custom datasets or use the reference UCI Oxford Parkinson's dataset, verify zero
subject-level data leakage, train multiple classical and ensemble algorithms, and persist validated artifacts.
""")

st.divider()

# ─── SECTION 1: Dataset Source & CSV Upload ───────────────────────────────────
st.markdown("### 1. Dataset Selection & Ingestion")

data_source = st.radio(
    "Choose Data Source:",
    options=["UCI Oxford Parkinson's Reference Dataset (195 recordings, 31 subjects)", "Upload Custom CSV File"],
    index=0,
)

df = None

if "UCI" in data_source:
    try:
        ds_file = DATA_DIR / "parkinsons.data"
        if not ds_file.exists():
            st.warning("Local parkinsons.data not found. Attempting download/copy...")
            from ml.dataset import download_dataset
            download_dataset()
        df = pd.read_csv(ds_file)
        st.success(f"Loaded Reference Dataset: **{len(df)} records** across **{df['name'].str.extract(r'(S\d+)', expand=False).nunique()} subjects**.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Uploaded dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

if df is None:
    st.info("Please load or upload a dataset to proceed.")
    st.stop()

with st.expander("Dataset Preview (First 5 Rows)", expanded=False):
    st.dataframe(df.head(5), use_container_width=True)

# ─── SECTION 2: Column Mapping ────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 2. Column Mapping")

all_cols = list(df.columns)

c_m1, c_m2 = st.columns(2)
with c_m1:
    subject_col = st.selectbox(
        "Subject Identifier Column (for Group-Aware Splitting):",
        options=all_cols,
        index=all_cols.index(SUBJECT_ID_COLUMN) if SUBJECT_ID_COLUMN in all_cols else 0,
        help="Recordings belonging to the same subject must never be split across train and test sets.",
    )
with c_m2:
    target_col = st.selectbox(
        "Target Label Column (0 = Healthy, 1 = Parkinson's):",
        options=all_cols,
        index=all_cols.index(TARGET_COLUMN) if TARGET_COLUMN in all_cols else (1 if len(all_cols) > 1 else 0),
    )

default_features = [c for c in all_cols if c not in [subject_col, target_col]]
feature_cols = st.multiselect(
    "Select Feature Columns for Model Training:",
    options=all_cols,
    default=default_features,
)

if not feature_cols:
    st.error("Please select at least one feature column.")
    st.stop()

# ─── SECTION 3: Dataset Quality Inspection ────────────────────────────────────
st.markdown("---")
st.markdown("### 3. Dataset Quality & Label Integrity Audit")

with st.spinner("Running dataset quality checks..."):
    # Extract subjects
    subjects = df[subject_col].astype(str)
    if subjects.str.contains(r"S\d+").any():
        subj_series = subjects.str.extract(r"(S\d+)", expand=False).fillna(subjects)
    else:
        subj_series = subjects

    total_rows = len(df)
    missing_count = int(df[feature_cols].isnull().sum().sum())
    n_subjects = subj_series.nunique()
    target_counts = df[target_col].value_counts().to_dict()

    # Subject consistency check
    temp_df = pd.DataFrame({"subject": subj_series, "target": df[target_col]})
    subject_labels = temp_df.groupby("subject")["target"].nunique()
    inconsistent = subject_labels[subject_labels > 1]

qc1, qc2, qc3, qc4 = st.columns(4)
with qc1:
    st.metric("Total Rows", total_rows)
with qc2:
    st.metric("Unique Subjects", n_subjects)
with qc3:
    st.metric("Missing Values", missing_count)
with qc4:
    st.metric("Inconsistent Subjects", len(inconsistent))

if len(inconsistent) > 0:
    st.error(f"Subject Label Inconsistency Detected: {list(inconsistent.index)} has contradictory labels across rows.")
else:
    st.success("Subject Label Consistency Verified: Every subject possesses a uniform clinical label across all recordings.")

col_dist1, col_dist2 = st.columns(2)
with col_dist1:
    st.markdown("**Target Class Distribution:**")
    st.write(target_counts)
with col_dist2:
    st.markdown("**Recordings Per Subject:**")
    st.write(f"Average: {total_rows / max(1, n_subjects):.1f} recordings/subject (Min: {temp_df.groupby('subject').size().min()}, Max: {temp_df.groupby('subject').size().max()})")

# ─── SECTION 4: Subject-Aware Splitting & Leakage Verification ────────────────
st.markdown("---")
st.markdown("### 4. Subject-Aware Train/Test Splitting")
st.markdown("""
To prevent **subject memorization leakage**, all recordings from a given subject are assigned
strictly to either the training set or test set.
""")

sp_col1, sp_col2 = st.columns(2)
with sp_col1:
    test_ratio = st.slider("Test Set Subject Ratio", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
with sp_col2:
    rand_seed = st.number_input("Random Split Seed", min_value=1, max_value=9999, value=42, step=1)

# Perform subject-aware split
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=int(rand_seed))
train_idx, test_idx = next(gss.split(df, df[target_col], groups=subj_series))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_subjs = set(subj_series.iloc[train_idx])
test_subjs = set(subj_series.iloc[test_idx])
overlap = train_subjs & test_subjs

c_ver1, c_ver2 = st.columns(2)
with c_ver1:
    if len(overlap) == 0:
        st.success(f"Zero Subject Leakage Confirmed:\n\n- Train Set: {len(train_df)} rows ({len(train_subjs)} subjects)\n- Test Set: {len(test_df)} rows ({len(test_subjs)} subjects)\n- Overlap: 0 subjects")
    else:
        st.error(f"Subject Leakage Detected: {overlap}")
with c_ver2:
    st.success("Zero Preprocessing Leakage Guaranteed:\n\nStandardScaler is fit strictly on X_train and transforms X_test using training parameters only.")

# ─── SECTION 5: Model Training & Comparative Evaluation ───────────────────────
st.markdown("---")
st.markdown("### 5. Multi-Algorithm Training & Evaluation")

available_models = get_model_configs()
selected_model_keys = st.multiselect(
    "Select Algorithms to Train & Benchmark:",
    options=list(available_models.keys()),
    default=list(available_models.keys()),
    format_func=lambda k: available_models[k]["name"],
)

if st.button("Run Training & Benchmarking", icon=":material/play_arrow:", type="primary"):
    with st.spinner("Preprocessing features and training candidate classifiers..."):
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X_train_raw = train_df[feature_cols].values
        X_test_raw = test_df[feature_cols].values
        y_train = train_df[target_col].values.astype(int)
        y_test = test_df[target_col].values.astype(int)

        # Impute if needed
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        # Scaler fit only on train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        training_results = {}
        for m_key in selected_model_keys:
            cfg = available_models[m_key]
            try:
                model, t_time = train_single_model(m_key, X_train_scaled, y_train, feature_cols)
                metrics = evaluate_model(model, X_test_scaled, y_test, model_name=cfg["name"])
                training_results[m_key] = {
                    "model": model,
                    "name": cfg["name"],
                    "description": cfg["description"],
                    "metrics": metrics,
                    "train_time_seconds": round(t_time, 3),
                }
            except Exception as e:
                training_results[m_key] = {"error": str(e), "name": cfg["name"]}

        st.session_state["model_lab_results"] = training_results
        st.session_state["model_lab_scaler"] = scaler
        st.session_state["model_lab_features"] = feature_cols
        st.session_state["model_lab_X_test"] = X_test_scaled
        st.session_state["model_lab_y_test"] = y_test
        st.session_state["model_lab_report"] = generate_evaluation_report(training_results, feature_cols)
        st.success("Training and benchmarking completed successfully.")

# Display Results if available
if "model_lab_results" in st.session_state:
    results = st.session_state["model_lab_results"]
    eval_rep = st.session_state["model_lab_report"]

    st.markdown("#### Model Performance Comparison Table")
    metrics_rows = []
    for m_key, r in results.items():
        if "error" in r:
            continue
        m = r["metrics"]
        metrics_rows.append({
            "Model": r["name"],
            "Accuracy": f"{m.get('accuracy', 0):.3f}",
            "Precision": f"{m.get('precision', 0):.3f}",
            "Recall (Sensitivity)": f"{m.get('recall', 0):.3f}",
            "Specificity": f"{m.get('specificity', 0):.3f}",
            "F1-Score": f"{m.get('f1', 0):.3f}",
            "ROC-AUC": f"{m.get('roc_auc', 0):.3f}",
            "PR-AUC": f"{m.get('pr_auc', 0):.3f}",
            "Train Latency": f"{r.get('train_time_seconds', 0):.3f}s",
        })

    df_metrics = pd.DataFrame(metrics_rows)
    st.dataframe(df_metrics, use_container_width=True)

    # Best model badge
    try:
        best_k, best_res = select_best_model(results)
        st.info(f"Top Composite Performer: **{best_res['name']}** (Selected based on balanced ROC-AUC, F1, and Sensitivity for clinical screening).")
    except Exception as e:
        best_k, best_res = list(results.keys())[0], list(results.values())[0]

    # Visualizations
    st.markdown("#### Benchmark Visualizations")
    col_v1, col_v2 = st.columns([3, 2])

    with col_v1:
        valid_comp_data = [d for d in eval_rep.get("comparison", []) if d.get("status") == "trained"]
        fig_bar = model_comparison_chart(valid_comp_data)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_v2:
        bm = best_res["metrics"]
        if "confusion_matrix" in bm:
            fig_cm = confusion_matrix_chart(bm["confusion_matrix"], model_name=best_res["name"])
            st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curves Overlay
    st.markdown("#### ROC Curve Analysis")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", line=dict(dash="dash", color="grey"), x0=0, x1=1, y0=0, y1=1)

    for m_key, r in results.items():
        if "error" not in r:
            roc_data = r["metrics"].get("roc_curve")
            if roc_data:
                fig_roc.add_trace(go.Scatter(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    mode="lines",
                    name=f"{r['name']} (AUC = {r['metrics']['roc_auc']:.2f})",
                ))

    fig_roc.update_layout(
        title="ROC Curves on Subject-Separated Test Set",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ─── SECTION 6: Experiment Persistence ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 6. Persist & Register Deployed Model")

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        model_to_save = st.selectbox(
            "Select Model to Save & Deploy:",
            options=list(results.keys()),
            format_func=lambda k: results[k]["name"],
            index=list(results.keys()).index(best_k) if best_k in results else 0,
        )
        save_version = st.text_input("Model Artifact Version Tag", value="v1.0.0")

    with c_p2:
        st.markdown("**Deployment Actions:**")
        if st.button("Save Artifacts & Register to Database", icon=":material/save:", type="primary"):
            sel_dict = results[model_to_save]
            art_dir = save_model_artifacts(
                model=sel_dict["model"],
                scaler=st.session_state["model_lab_scaler"],
                feature_names=st.session_state["model_lab_features"],
                metrics=sel_dict["metrics"],
                model_key=model_to_save,
                version=save_version,
            )

            # Register in ModelRegistry table
            session = get_session()
            try:
                existing = session.query(ModelRegistry).filter(ModelRegistry.version == save_version).first()
                if not existing:
                    reg = ModelRegistry(
                        model_name="pd_voice_classifier",
                        version=save_version,
                        model_type=model_to_save,
                        status=ModelStatus.VALIDATED,
                        dataset_name=data_source,
                        training_date=datetime.now(timezone.utc),
                        feature_schema=st.session_state["model_lab_features"],
                        preprocessing_version="1.0",
                        metrics={k: round(v, 4) if isinstance(v, float) else v for k, v in sel_dict["metrics"].items() if k not in ["confusion_matrix", "classification_report_dict"]},
                        validation_protocol="subject-aware group shuffle split",
                        model_file_path=str(art_dir / "model.joblib"),
                        scaler_file_path=str(art_dir / "scaler.joblib"),
                        metadata_file_path=str(art_dir / "metadata.json"),
                        intended_use="AI-assisted voice biomarker screening for Parkinson's disease",
                        known_limitations="Trained on UCI dataset (31 subjects). Not for definitive clinical diagnosis.",
                    )
                    session.add(reg)
                    session.commit()
                st.success(f"Successfully saved artifacts and registered model {save_version}.")
            except Exception as e:
                session.rollback()
                st.error(f"Failed to record in ModelRegistry: {e}")
            finally:
                session.close()

    # Export report JSON
    rep_json = json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(df),
        "test_subjects": len(test_subjs),
        "train_subjects": len(train_subjs),
        "results": {
            k: {
                "name": v["name"],
                "metrics": {
                    mk: round(mv, 4) if isinstance(mv, float) else mv
                    for mk, mv in v["metrics"].items()
                    if mk not in ["confusion_matrix", "classification_report_dict", "roc_curve", "pr_curve"]
                },
                "latency_sec": v["train_time_seconds"],
            }
            for k, v in results.items() if "error" not in v
        },
    }, indent=2)

    st.download_button(
        "Download Experiment Benchmark (JSON)",
        icon=":material/download:",
        data=rep_json,
        file_name=f"model_benchmark_{save_version}.json",
        mime="application/json",
    )
