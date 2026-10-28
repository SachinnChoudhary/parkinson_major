"""
Start Screening — Multi-step screening workflow.

This is the primary patient-facing workflow. It guides users through:
1. Consent and Introduction
2. Patient Information
3. Natural Language Symptom Description
4. Structured Symptom Questionnaire
5. Optional Voice Biomarker Analysis
6. Review & Submit
7. Results (Honest results: real ML inference if voice data provided; clinical symptom summary if not)
"""

import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    APP_NAME, MEDICAL_DISCLAIMER, MOTOR_SYMPTOMS, NON_MOTOR_SYMPTOMS,
    SEVERITY_LEVELS, VOICE_FEATURES,
)
from database.engine import init_db, get_session
from database.models import Screening, ScreeningStatus, Prediction
from components.auth_manager import init_session_state
from nlp.symptom_extractor import extract_symptoms_keyword, extract_time_info
from ml.inference import get_inference_service
from ml.explainability import explain_prediction
from components.charts import shap_waterfall_chart, risk_gauge_chart
from components.report_generator import generate_screening_report

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"Start Screening — {APP_NAME}",
    layout="wide",
)

init_db()
init_session_state()

# ─── Session State Initialization ─────────────────────────────────────────────

if "screening_step" not in st.session_state:
    st.session_state.screening_step = 1

if "screening_data" not in st.session_state:
    st.session_state.screening_data = {
        "session_id": str(uuid.uuid4()),
        "consent_given": False,
        "age_range": None,
        "symptom_duration": None,
        "existing_diagnosis": None,
        "family_history": None,
        "natural_language_input": "",
        "extracted_symptoms": [],
        "user_reviewed_symptoms": [],
        "questionnaire_responses": {},
        "voice_features": None,
        "voice_source": "none",
    }

if "screening_result" not in st.session_state:
    st.session_state.screening_result = None

data = st.session_state.screening_data

TOTAL_STEPS = 7
step = st.session_state.screening_step

st.markdown(f"### Screening Assessment — Step {step} of {TOTAL_STEPS}")
st.progress(step / TOTAL_STEPS)


def next_step():
    st.session_state.screening_step += 1


def prev_step():
    st.session_state.screening_step = max(1, st.session_state.screening_step - 1)


def go_to_step(s):
    st.session_state.screening_step = s


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Consent & Introduction
# ═══════════════════════════════════════════════════════════════════════════════

if step == 1:
    st.markdown("## Introduction & Informed Consent")

    st.markdown("""
    Welcome to the PD-Aware screening assessment. Before proceeding, please review the
    following clinical and operational parameters carefully.

    ### Clinical Objectives
    This system evaluates patient-reported observations and biomedical acoustic data to detect
    statistical indicators associated with Parkinson's disease.
    **This software does not diagnose or rule out Parkinson's disease.**

    ### Data Handled
    - Self-reported age bracket and clinical context (optional)
    - Qualitative symptom descriptions and standardized questionnaire ratings
    - Voice biomedical acoustic measurements (optional)

    ### Security & Privacy Protections
    - Data analysis executes strictly on this local system
    - No identifiable personal health information is transmitted to third-party services
    - Results are archived locally only upon explicit submission
    """)

    st.markdown(f"""
    <div style="background: #FEF9C3; border: 1px solid #FCD34D;
    border-radius: 10px; padding: 1rem 1.5rem; margin: 1rem 0; font-size: 0.85rem;
    color: #92400E;">
        <strong>Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)

    consent = st.checkbox(
        "I understand that this is an informational decision-support tool, not a diagnostic medical device, "
        "and that any findings must be evaluated by a licensed healthcare professional.",
        key="consent_checkbox",
        value=data.get("consent_given", False),
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Continue", icon=":material/arrow_forward:", type="primary", disabled=not consent):
            data["consent_given"] = True
            data["consent_timestamp"] = datetime.now(timezone.utc).isoformat()
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Patient Information
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 2:
    st.markdown("## Clinical Context & Demographics")
    st.markdown(
        "Please provide context information. All fields are optional but provide useful "
        "baseline information for clinical review."
    )

    col1, col2 = st.columns(2)

    with col1:
        data["age_range"] = st.selectbox(
            "Age Range",
            options=["Prefer not to say", "Under 30", "30-39", "40-49",
                     "50-59", "60-69", "70-79", "80+"],
            index=0 if not data.get("age_range") else ["Prefer not to say", "Under 30", "30-39", "40-49",
                     "50-59", "60-69", "70-79", "80+"].index(data["age_range"]),
            key="age_range",
        )

        data["symptom_duration"] = st.selectbox(
            "Symptom Duration",
            options=["Not applicable", "Less than 1 month", "1-6 months",
                     "6-12 months", "1-2 years", "More than 2 years"],
            index=0 if not data.get("symptom_duration") else ["Not applicable", "Less than 1 month", "1-6 months",
                     "6-12 months", "1-2 years", "More than 2 years"].index(data["symptom_duration"]),
            key="symptom_duration",
        )

    with col2:
        data["existing_diagnosis"] = st.selectbox(
            "Prior Parkinson's Diagnosis",
            options=["No", "Yes", "Under investigation", "Prefer not to say"],
            index=0 if not data.get("existing_diagnosis") else ["No", "Yes", "Under investigation", "Prefer not to say"].index(data["existing_diagnosis"]),
            key="existing_diagnosis",
        )

        data["family_history"] = st.selectbox(
            "Family History of Movement Disorders",
            options=["None known", "Yes — immediate family",
                     "Yes — extended family", "Unsure", "Prefer not to say"],
            index=0 if not data.get("family_history") else ["None known", "Yes — immediate family",
                     "Yes — extended family", "Unsure", "Prefer not to say"].index(data["family_history"]),
            key="family_history",
        )

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Back", icon=":material/arrow_back:"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue", icon=":material/arrow_forward:", type="primary"):
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Natural Language Symptom Description
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 3:
    st.markdown("## Patient Narrative")
    st.markdown(
        "Please describe any physical changes, movement difficulties, or concerns in your own words. "
        "The system extracts candidate clinical indicators for your **explicit verification** before inclusion."
    )

    st.markdown(
        '*Example: "I have noticed shaking in my right hand when it is resting. '
        'My handwriting has become smaller, and I feel slower when walking."*'
    )

    text_input = st.text_area(
        "Narrative Description (Optional)",
        value=data.get("natural_language_input", ""),
        height=140,
        key="nl_input",
        placeholder="Document any changes in movement, handwriting, balance, speech, or sleep...",
    )

    data["natural_language_input"] = text_input

    if text_input.strip():
        st.markdown("---")
        st.markdown("### Review Extracted Indicators")
        st.markdown(
            "The natural-language extraction pipeline flagged the following indicators. "
            "Please confirm or uncheck each item based on your actual experience."
        )

        extracted = extract_symptoms_keyword(text_input)
        time_info = extract_time_info(text_input)

        if extracted:
            reviewed = []
            for i, symptom in enumerate(extracted):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.markdown(f"**{symptom['name']}** — {symptom['description']}")
                    if symptom.get("context"):
                        st.caption(f'Context: "{symptom["context"]}"')
                with c2:
                    st.markdown(f"Category: `{symptom['category'].title()}`")
                with c3:
                    keep = st.checkbox("Include", value=True, key=f"keep_sym_{i}")

                if keep:
                    reviewed.append(symptom)

            data["extracted_symptoms"] = extracted
            data["user_reviewed_symptoms"] = reviewed
        else:
            st.info("No specific symptom keywords were identified from the text. "
                    "You may specify symptoms directly in the structured questionnaire in the next step.")

        if time_info:
            st.markdown("#### Mentioned Timing Details")
            for category, matches in time_info.items():
                st.caption(f"**{category.title()}:** {', '.join(matches)}")
    else:
        st.info("Note: This qualitative description step is optional. You may proceed directly to the questionnaire.")

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Back", icon=":material/arrow_back:"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue", icon=":material/arrow_forward:", type="primary"):
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Structured Symptom Questionnaire
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 4:
    st.markdown("## Standardized Symptom Questionnaire")
    st.markdown(
        "Please indicate whether you have experienced any of the following motor and non-motor signs. "
        "Select the severity rating that most accurately reflects your daily experience."
    )

    responses = data.get("questionnaire_responses", {})

    st.markdown("### Motor (Movement) Indicators")
    for symptom_id, symptom_name, symptom_desc in MOTOR_SYMPTOMS:
        st.markdown(f"**{symptom_name}**")
        st.caption(symptom_desc)
        responses[symptom_id] = st.select_slider(
            f"Severity of {symptom_name}",
            options=[s[1] for s in SEVERITY_LEVELS],
            value=responses.get(symptom_id, "Not present"),
            key=f"q_{symptom_id}",
            label_visibility="collapsed",
        )
        st.markdown("")

    st.divider()

    st.markdown("### Non-Motor Indicators")
    for symptom_id, symptom_name, symptom_desc in NON_MOTOR_SYMPTOMS:
        st.markdown(f"**{symptom_name}**")
        st.caption(symptom_desc)
        if symptom_id == "other":
            responses[symptom_id] = st.text_input(
                "Other symptoms",
                value=responses.get(symptom_id, ""),
                key=f"q_{symptom_id}",
                placeholder="Describe any other symptoms or notes...",
            )
        else:
            responses[symptom_id] = st.select_slider(
                f"Severity of {symptom_name}",
                options=[s[1] for s in SEVERITY_LEVELS],
                value=responses.get(symptom_id, "Not present"),
                key=f"q_{symptom_id}",
                label_visibility="collapsed",
            )
        st.markdown("")

    data["questionnaire_responses"] = responses

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Back", icon=":material/arrow_back:"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue", icon=":material/arrow_forward:", type="primary"):
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Optional Voice Biomarker Analysis
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 5:
    st.markdown("## Voice Acoustic Biomarker Analysis (Optional)")
    st.markdown("""
    The deployed machine learning inference engine evaluates **22 micro-acoustic features**
    (including pitch frequency, frequency jitter, amplitude shimmer, noise-to-harmonics ratio, and recurrence entropy)
    derived from sustained phonation.

    **Note on Modality Separation:** Clinical questionnaires and acoustic features represent distinct data streams.
    The automated ML model evaluates acoustic measurements. If voice data is omitted, an organized clinical summary
    is generated without an artificial prediction.
    """)

    voice_option = st.radio(
        "Select Acoustic Analysis Pathway:",
        options=[
            "Skip voice analysis (Proceed with symptom-only clinical summary)",
            "Load verified dataset reference profile (Benchmarking case)",
            "Upload acoustic features CSV file",
            "Manually enter acoustic parameters",
        ],
        index=0 if data.get("voice_source") == "none" else 1,
    )

    if "Skip" in voice_option:
        data["voice_features"] = None
        data["voice_source"] = "none"
        st.info("You selected to skip voice analysis. A comprehensive clinical summary of your reported symptoms will be compiled without an automated ML risk prediction.")

    elif "Load verified" in voice_option:
        st.markdown("### Reference Case Selection")
        sample_choice = st.selectbox(
            "Select reference profile from UCI dataset:",
            options=[
                "Healthy Control Reference (Subject S07 — Normal pitch periodicity & low jitter)",
                "Parkinson's-Associated Reference (Subject S01 — Elevated jitter, shimmer & entropy)",
            ],
        )

        try:
            from config.settings import DATA_DIR
            ds_path = DATA_DIR / "parkinsons.data"
            if ds_path.exists():
                df_samples = pd.read_csv(ds_path)
                if "Healthy" in sample_choice:
                    sample_row = df_samples[df_samples["status"] == 0].iloc[0]
                    src_label = "Verified Healthy Reference Case (S07)"
                else:
                    sample_row = df_samples[df_samples["status"] == 1].iloc[0]
                    src_label = "Verified Parkinson's Reference Case (S01)"

                feat_dict = {f: float(sample_row[f]) for f in VOICE_FEATURES if f in sample_row}
                data["voice_features"] = feat_dict
                data["voice_source"] = src_label

                st.success(f"Loaded: {src_label} ({len(feat_dict)} acoustic features verified).")
                with st.expander("View Acoustic Parameters", expanded=False):
                    st.dataframe(pd.DataFrame([feat_dict]).T.rename(columns={0: "Value"}))
            else:
                st.error("Dataset reference file not found in local data directory.")
        except Exception as e:
            st.error(f"Failed to load sample: {e}")

    elif "Upload acoustic features CSV" in voice_option:
        st.markdown("### File Ingestion")
        uploaded_file = st.file_uploader(
            "Upload CSV containing single-row acoustic parameters",
            type=["csv"],
        )
        if uploaded_file is not None:
            try:
                up_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded record:", up_df.head(1))
                matched_feats = {}
                missing_feats = []
                for f in VOICE_FEATURES:
                    if f in up_df.columns:
                        matched_feats[f] = float(up_df[f].iloc[0])
                    else:
                        missing_feats.append(f)

                if missing_feats:
                    st.warning(f"Note: {len(missing_feats)} expected features not found in uploaded file. Missing: {missing_feats[:3]}...")
                if matched_feats:
                    data["voice_features"] = matched_feats
                    data["voice_source"] = f"User Upload ({uploaded_file.name})"
                    st.success(f"Successfully loaded {len(matched_feats)} features from CSV.")
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")

    elif "Manually enter" in voice_option:
        st.markdown("### Direct Acoustic Parameter Input")
        st.caption("Input measured acoustic variables. Defaults reflect typical healthy baselines.")
        c1, c2, c3 = st.columns(3)
        with c1:
            fo = st.number_input("Average F0 (Hz)", min_value=50.0, max_value=300.0, value=150.0, step=1.0)
            fhi = st.number_input("Max F0 (Hz)", min_value=50.0, max_value=400.0, value=180.0, step=1.0)
            flo = st.number_input("Min F0 (Hz)", min_value=40.0, max_value=300.0, value=120.0, step=1.0)
            jitter_pct = st.number_input("Jitter (%)", min_value=0.0, max_value=0.1, value=0.005, format="%.5f")
        with c2:
            shimmer = st.number_input("Shimmer", min_value=0.0, max_value=0.2, value=0.03, format="%.4f")
            shimmer_db = st.number_input("Shimmer (dB)", min_value=0.0, max_value=2.0, value=0.3, format="%.3f")
            nhr = st.number_input("Noise-to-Harmonics (NHR)", min_value=0.0, max_value=1.0, value=0.02, format="%.4f")
            hnr = st.number_input("Harmonics-to-Noise (HNR)", min_value=5.0, max_value=40.0, value=22.0, step=0.5)
        with c3:
            rpde = st.number_input("RPDE", min_value=0.1, max_value=1.0, value=0.5, format="%.4f")
            dfa = st.number_input("DFA", min_value=0.1, max_value=1.0, value=0.7, format="%.4f")
            spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=-5.0, format="%.3f")
            ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.2, format="%.4f")

        manual_dict = {
            "MDVP:Fo(Hz)": fo, "MDVP:Fhi(Hz)": fhi, "MDVP:Flo(Hz)": flo,
            "MDVP:Jitter(%)": jitter_pct, "MDVP:Jitter(Abs)": jitter_pct / 1000,
            "MDVP:RAP": jitter_pct * 0.5, "MDVP:PPQ": jitter_pct * 0.6, "Jitter:DDP": jitter_pct * 1.5,
            "MDVP:Shimmer": shimmer, "MDVP:Shimmer(dB)": shimmer_db,
            "Shimmer:APQ3": shimmer * 0.5, "Shimmer:APQ5": shimmer * 0.6,
            "MDVP:APQ": shimmer * 0.8, "Shimmer:DDA": shimmer * 1.5,
            "NHR": nhr, "HNR": hnr, "RPDE": rpde, "DFA": dfa,
            "spread1": spread1, "spread2": 0.2, "D2": 2.3, "PPE": ppe,
        }
        data["voice_features"] = manual_dict
        data["voice_source"] = "Manual User Entry"
        st.success("Manual acoustic parameters registered.")

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Back", icon=":material/arrow_back:"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Continue to Review", icon=":material/arrow_forward:", type="primary"):
            next_step()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Review & Submit
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 6:
    st.markdown("## Review Case Information")
    st.markdown("Please verify the recorded parameters prior to final generation.")

    with st.expander("Patient Information", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Age Bracket:** {data.get('age_range', 'Not specified')}")
            st.markdown(f"**Symptom Duration:** {data.get('symptom_duration', 'Not specified')}")
        with c2:
            st.markdown(f"**Prior Diagnosis:** {data.get('existing_diagnosis', 'None')}")
            st.markdown(f"**Family History:** {data.get('family_history', 'None')}")

    if data.get("natural_language_input"):
        with st.expander("Patient Description", expanded=True):
            st.markdown(f'> "{data["natural_language_input"]}"')
            reviewed = data.get("user_reviewed_symptoms", [])
            if reviewed:
                st.markdown("**Confirmed Clinical Signs:**")
                for s in reviewed:
                    st.markdown(f"- **{s['name']}** ({s['category']})")

    responses = data.get("questionnaire_responses", {})
    reported = {k: v for k, v in responses.items() if v and v not in ("Not present", "")}
    with st.expander("Questionnaire Responses", expanded=True):
        if reported:
            for key, value in reported.items():
                display_name = key.replace("_", " ").title()
                for items in [MOTOR_SYMPTOMS, NON_MOTOR_SYMPTOMS]:
                    for sid, sname, _ in items:
                        if sid == key:
                            display_name = sname
                            break
                st.markdown(f"- **{display_name}:** {value}")
        else:
            st.info("No specific symptoms reported in questionnaire.")

    with st.expander("Acoustic Biomarker Status", expanded=True):
        if data.get("voice_features"):
            st.success(f"Acoustic features attached: {data.get('voice_source')} ({len(data['voice_features'])} features)")
        else:
            st.info("No voice acoustic data attached. Screening summary will document reported symptoms without an automated ML risk prediction.")

    st.markdown(f"**Consent Verification:** {'[Confirmed]' if data.get('consent_given') else '[Pending]'}")

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Back", icon=":material/arrow_back:"):
            prev_step()
            st.rerun()
    with col2:
        if st.button("Submit Screening", icon=":material/check_circle:", type="primary"):
            session = get_session()
            try:
                screening = Screening(
                    session_id=data["session_id"],
                    user_id=st.session_state.get("user", {}).get("id") if st.session_state.get("user") else None,
                    status=ScreeningStatus.COMPLETED,
                    age_range=data.get("age_range"),
                    symptom_duration=data.get("symptom_duration"),
                    existing_diagnosis=data.get("existing_diagnosis"),
                    family_history=data.get("family_history"),
                    natural_language_input=data.get("natural_language_input"),
                    extracted_symptoms=data.get("extracted_symptoms"),
                    user_reviewed_symptoms=data.get("user_reviewed_symptoms"),
                    questionnaire_responses=data.get("questionnaire_responses"),
                    voice_features=data.get("voice_features"),
                    consent_given=data.get("consent_given", False),
                    consent_timestamp=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )
                session.add(screening)
                session.commit()
                st.session_state.screening_db_id = screening.id
                logger.info(f"Screening saved with id: {screening.id}")

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save screening: {e}")
            finally:
                session.close()

            next_step()
            st.rerun()

    with col3:
        if st.button("Reset Form", icon=":material/restart_alt:"):
            st.session_state.screening_step = 1
            st.session_state.screening_data = {
                "session_id": str(uuid.uuid4()),
                "consent_given": False,
                "voice_features": None,
                "voice_source": "none",
            }
            st.session_state.screening_result = None
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: Results
# ═══════════════════════════════════════════════════════════════════════════════

elif step == 7:
    st.markdown("## Screening Results & Clinical Assessment")

    voice_features = data.get("voice_features")

    # CASE A: Voice data provided -> Run real ML inference
    if voice_features:
        service = get_inference_service()

        if not service.is_loaded:
            st.warning(
                "Model Artifacts Not Available: The machine learning model is currently not loaded. "
                "Please run model training in the Model Lab to generate active artifacts."
            )
        else:
            result = service.predict(voice_features)

            if result and result.get("success"):
                st.session_state.screening_result = result

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("### Model Assessment")
                    category = result.get("prediction_category", "unknown")
                    score = result.get("prediction_score")

                    if score is not None:
                        fig = risk_gauge_chart(score, category)
                        st.plotly_chart(fig, use_container_width=True)

                    if category == "parkinson":
                        st.markdown(
                            "The model identified acoustic patterns in the voice data "
                            "that were **more consistent** with its Parkinson's-associated training examples."
                        )
                    else:
                        st.markdown(
                            "The model did **not** identify strong acoustic patterns associated "
                            "with Parkinson's disease in the evaluated measurements."
                        )

                    st.markdown(
                        "*This result is not a diagnosis and cannot establish whether you have Parkinson's disease.*"
                    )

                with col2:
                    st.markdown("### Model Provenance")
                    st.markdown(f"**Data Source:** {data.get('voice_source', 'Provided features')}")
                    st.markdown(f"**Model Type:** {result.get('model_type', 'N/A').replace('_', ' ').title()}")
                    st.markdown(f"**Model Version:** {result.get('model_version', 'N/A')}")
                    st.markdown(f"**Inference Latency:** {result.get('inference_time_ms', 0):.1f} ms")

                    if result.get("issues"):
                        st.markdown("**Validation Notes:**")
                        for issue in result["issues"]:
                            st.caption(f"- {issue}")

                # SHAP Feature Explanations
                st.divider()
                st.markdown("### Contributing Factors (Acoustic Biomarkers)")
                explanation = None
                try:
                    feat_array = np.array([[voice_features.get(f, 0.0) for f in service.expected_features]])
                    explanation = explain_prediction(service.model, service.scaler, feat_array, service.expected_features)

                    if explanation.get("success"):
                        fig = shap_waterfall_chart(explanation["contributions"])
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(explanation.get("note", ""))

                        with st.expander("Detailed Feature Contributions", expanded=False):
                            for contrib in explanation.get("contributions", []):
                                direction_tag = "[Increases Score]" if contrib["shap_value"] > 0 else "[Decreases Score]"
                                st.markdown(
                                    f"**{contrib['description']}**: "
                                    f"`{direction_tag}` (SHAP Impact: {contrib['shap_value']:+.4f})"
                                )
                except Exception as e:
                    logger.error(f"Explanation error: {e}")

                # Save prediction to DB if screening was saved
                if st.session_state.get("screening_db_id"):
                    session = get_session()
                    try:
                        existing_pred = session.query(Prediction).filter(
                            Prediction.screening_id == st.session_state.screening_db_id
                        ).first()
                        if not existing_pred:
                            pred_record = Prediction(
                                screening_id=st.session_state.screening_db_id,
                                model_version=result.get("model_version", "v1.0.0"),
                                model_type=result.get("model_type", "gradient_boosting"),
                                feature_schema_version=result.get("feature_schema_version", "1.0"),
                                preprocessing_version=result.get("preprocessing_version", "1.0"),
                                prediction_label=result.get("prediction_label", 0),
                                prediction_score=result.get("prediction_score"),
                                prediction_category=result.get("prediction_category"),
                                input_features=voice_features,
                                feature_names=service.expected_features,
                                shap_values=explanation.get("shap_values") if explanation else None,
                                inference_time_ms=result.get("inference_time_ms"),
                            )
                            session.add(pred_record)
                            session.commit()
                    except Exception as e:
                        session.rollback()
                        logger.error(f"Failed to save prediction record: {e}")
                    finally:
                        session.close()

            elif result:
                st.error(f"Inference failed: {result.get('error')}")

    # CASE B: Voice data was NOT provided -> Honest summary without fake prediction
    else:
        st.info(
            "Clinical Notice: No Automated Prediction Generated\n\n"
            "The current validated machine learning model requires acoustic voice measurements "
            "(e.g., vocal fundamental frequency, jitter, shimmer, harmonics-to-noise ratio) and "
            "cannot evaluate clinical symptoms alone. To maintain clinical integrity, "
            "no arbitrary or simulated prediction has been generated.\n\n"
            "Your reported symptoms have been organized into the structured clinical summary below."
        )

    # Symptom Summary Display (Always shown)
    st.divider()
    st.markdown("### Reported Symptoms Summary")

    responses = data.get("questionnaire_responses", {})
    reported = {k: v for k, v in responses.items() if v and v not in ("Not present", "")}

    if reported:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Movement-Related (Motor) Signs:**")
            motor_reported = False
            for sid, sname, _ in MOTOR_SYMPTOMS:
                if sid in reported:
                    st.markdown(f"- **{sname}:** {reported[sid]}")
                    motor_reported = True
            if not motor_reported:
                st.caption("None reported.")

        with c2:
            st.markdown("**Non-Motor Signs:**")
            non_motor_reported = False
            for sid, sname, _ in NON_MOTOR_SYMPTOMS:
                if sid in reported:
                    st.markdown(f"- **{sname}:** {reported[sid]}")
                    non_motor_reported = True
            if not non_motor_reported:
                st.caption("None reported.")
    else:
        st.write("No specific symptoms were reported.")

    if data.get("user_reviewed_symptoms"):
        st.markdown("**Extracted from Narrative:**")
        for s in data["user_reviewed_symptoms"]:
            st.markdown(f"- {s['name']} ({s['category']})")

    # Limitations & Guidance
    st.divider()
    st.markdown("### Recommended Clinical Follow-Up")
    st.markdown("""
    - **Discuss with your doctor:** If you or a family member notice persistent involuntary movement,
      muscle stiffness, slowing of gait, or other neurological changes, consult a licensed physician or neurologist.
    - **Download the screening report:** Use the button below to download a PDF copy of this
      summary to share with your healthcare provider.
    - **Not a diagnosis:** This screening tool does not replace a physical clinical examination.
    """)

    # PDF Download
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        try:
            report_id = f"PDA-{data['session_id'][:8].upper()}"
            pred_data = st.session_state.get("screening_result") if voice_features else None
            exp_data = explanation if voice_features and explanation and explanation.get("success") else None

            pdf_bytes = generate_screening_report(
                report_id=report_id,
                screening_data=data,
                prediction_data=pred_data,
                explanation_data=exp_data,
                include_symptoms=True,
                include_prediction=bool(voice_features and pred_data),
                include_explanation=bool(voice_features and exp_data),
            )
            st.download_button(
                "Download Clinical Report (PDF)",
                icon=":material/download:",
                data=bytes(pdf_bytes),
                file_name=f"pd_aware_report_{report_id}.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as e:
            logger.error(f"PDF download error: {e}")
            st.error(f"Error generating PDF: {e}")

    with col2:
        if st.button("Start New Screening", icon=":material/add_circle:"):
            st.session_state.screening_step = 1
            st.session_state.screening_data = {
                "session_id": str(uuid.uuid4()),
                "consent_given": False,
                "voice_features": None,
                "voice_source": "none",
            }
            st.session_state.screening_result = None
            st.rerun()
