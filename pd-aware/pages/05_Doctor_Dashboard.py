"""
Doctor Dashboard — Clinical review portal for healthcare professionals.

Allows authorized clinicians to:
- Review patient-submitted screenings and symptom histories
- Inspect machine learning outputs and SHAP feature attributions
- Record separate clinical notes and observations
- Export professional screening summaries (PDF)
- Enforce role-based access control and audit logging
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import APP_NAME, MEDICAL_DISCLAIMER, MOTOR_SYMPTOMS, NON_MOTOR_SYMPTOMS
from database.engine import init_db, get_session
from database.models import User, UserRole, Screening, Prediction, ClinicianNote, AuditLog
from components.auth_manager import (
    init_session_state, login_form, authenticate_user, register_user,
)
from components.report_generator import generate_screening_report
from components.charts import shap_waterfall_chart, risk_gauge_chart
from ml.explainability import explain_prediction
from ml.inference import get_inference_service

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=f"Doctor Dashboard — {APP_NAME}",
    layout="wide",
)

init_db()
init_session_state()

st.markdown("## Clinician Review Portal")
st.markdown("""
This interface is designed for licensed physicians and neurologists to review
patient-submitted screening records, evaluate acoustic voice model indicators,
and document separate clinical observations.
""")

# ─── Auth Verification ────────────────────────────────────────────────────────
is_authenticated = st.session_state.get("authenticated", False)
current_user = st.session_state.get("user", {})
user_role = current_user.get("role") if is_authenticated else None

if not is_authenticated or user_role not in [UserRole.DOCTOR.value, UserRole.ADMIN.value]:
    st.warning("Clinician Authentication Required: This portal accesses medical records and requires authorized clinician credentials.")

    tab_login, tab_demo = st.tabs(["Clinician Sign In", "Demo Clinician Access"])

    with tab_login:
        c1, c2 = st.columns([1, 1])
        with c1:
            u_name = st.text_input("Username", key="doc_login_u")
            u_pass = st.text_input("Password", type="password", key="doc_login_p")
            if st.button("Sign In", icon=":material/login:", type="primary", key="btn_doc_signin"):
                success, u_info = authenticate_user(u_name, u_pass)
                if success:
                    if u_info.get("role") in [UserRole.DOCTOR.value, UserRole.ADMIN.value]:
                        st.session_state.authenticated = True
                        st.session_state.user = u_info
                        st.success("Authenticated as Clinician.")
                        st.rerun()
                    else:
                        st.error("Account does not have Clinician or Administrator privileges.")
                else:
                    st.error("Invalid credentials.")

    with tab_demo:
        st.info("For evaluation and demonstration purposes, select below to access with a pre-configured clinician profile.")
        if st.button("Log In as Dr. Sharma (Neurologist Demo)", icon=":material/badge:", type="secondary"):
            session = get_session()
            doc = session.query(User).filter(User.username == "dr_sharma").first()
            if not doc:
                register_user(
                    username="dr_sharma",
                    email="dr.sharma@hospital.org",
                    password="Password123!",
                    full_name="Dr. R. Sharma, MD (Neurology)",
                    role=UserRole.DOCTOR,
                )
            session.close()

            success, u_info = authenticate_user("dr_sharma", "Password123!")
            if success:
                st.session_state.authenticated = True
                st.session_state.user = u_info
                st.rerun()

    st.stop()

# ─── Authenticated Clinician Header ───────────────────────────────────────────
c_header1, c_header2 = st.columns([3, 1])
with c_header1:
    st.success(f"Signed in as: **{current_user.get('full_name')}** ({current_user.get('role').title()})")
with c_header2:
    if st.button("Sign Out", icon=":material/logout:"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

st.divider()

# ─── Load Screenings ──────────────────────────────────────────────────────────
session = get_session()
try:
    screenings = session.query(Screening).order_by(Screening.created_at.desc()).all()
    records_data = []
    for s in screenings:
        pred = s.prediction
        records_data.append({
            "Database ID": s.id,
            "Session ID": s.session_id[:8] + "...",
            "Created": s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "N/A",
            "Age Range": s.age_range or "Not stated",
            "Symptom Duration": s.symptom_duration or "N/A",
            "Existing Dx": s.existing_diagnosis or "No",
            "Voice Analyzed": "Yes" if s.voice_features else "No",
            "ML Result": pred.prediction_category.title() if pred and pred.prediction_category else "No voice data",
            "ML Score": f"{pred.prediction_score:.1%}" if pred and pred.prediction_score is not None else "N/A",
        })
except Exception as e:
    logger.error(f"Error loading screenings: {e}")
    records_data = []
finally:
    session.close()

st.markdown("### Patient Screening Registry")

if not records_data:
    st.info("No patient screening sessions recorded yet. Completed screenings from 'Start Screening' will appear here.")
    st.stop()

df_screenings = pd.DataFrame(records_data)

# Filter controls
col_f1, col_f2 = st.columns(2)
with col_f1:
    filter_voice = st.selectbox("Filter by Voice Data", options=["All", "Voice Analyzed", "No Voice Data"])
with col_f2:
    search_id = st.text_input("Search by Session ID or Age Bracket", placeholder="e.g., 60-69")

filtered_df = df_screenings.copy()
if filter_voice == "Voice Analyzed":
    filtered_df = filtered_df[filtered_df["Voice Analyzed"] == "Yes"]
elif filter_voice == "No Voice Data":
    filtered_df = filtered_df[filtered_df["Voice Analyzed"] == "No"]

if search_id:
    filtered_df = filtered_df[
        filtered_df["Session ID"].str.contains(search_id, case=False) |
        filtered_df["Age Range"].str.contains(search_id, case=False)
    ]

st.dataframe(filtered_df, use_container_width=True)

# ─── Detailed Record Inspection ───────────────────────────────────────────────
st.divider()
st.markdown("### Case Detail & Clinical Notes")

selected_db_id = st.selectbox(
    "Select screening record to review:",
    options=filtered_df["Database ID"].tolist(),
    format_func=lambda x: f"ID: {x[:10]}... | {df_screenings.loc[df_screenings['Database ID'] == x, 'Created'].values[0]} | Age: {df_screenings.loc[df_screenings['Database ID'] == x, 'Age Range'].values[0]} | ML: {df_screenings.loc[df_screenings['Database ID'] == x, 'ML Result'].values[0]}",
)

if selected_db_id:
    session = get_session()
    try:
        screening = session.query(Screening).filter(Screening.id == selected_db_id).first()
        if screening:
            # Audit log view action
            view_log = AuditLog(
                user_id=current_user.get("id"),
                action="view_patient_screening",
                resource_type="screening",
                resource_id=screening.id,
                details={"viewer": current_user.get("username")},
            )
            session.add(view_log)
            session.commit()

            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("#### Patient Context")
                st.write(f"**Session ID:** `{screening.session_id}`")
                st.write(f"**Screening Timestamp:** {screening.created_at.strftime('%Y-%m-%d %H:%M UTC') if screening.created_at else 'N/A'}")
                st.write(f"**Age Range:** {screening.age_range or 'Not specified'}")
                st.write(f"**Reported Duration:** {screening.symptom_duration or 'Not specified'}")
                st.write(f"**Prior Diagnosis:** {screening.existing_diagnosis or 'No'}")
                st.write(f"**Family History:** {screening.family_history or 'None known'}")

                if screening.natural_language_input:
                    st.markdown("#### Patient Description")
                    st.info(f'"{screening.natural_language_input}"')

            with c2:
                st.markdown("#### Reported Questionnaire Symptoms")
                q_resp = screening.questionnaire_responses or {}
                reported_items = {k: v for k, v in q_resp.items() if v and v not in ("Not present", "")}
                if reported_items:
                    for k, v in reported_items.items():
                        st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")
                else:
                    st.caption("No positive symptoms reported in questionnaire.")

            # Model prediction inspection
            st.markdown("---")
            st.markdown("#### Machine Learning Assessment")
            pred = screening.prediction

            if pred and screening.voice_features:
                mc1, mc2 = st.columns(2)
                with mc1:
                    score = pred.prediction_score
                    cat = pred.prediction_category or "unknown"
                    if score is not None:
                        fig = risk_gauge_chart(score, cat)
                        st.plotly_chart(fig, use_container_width=True)
                    st.write(f"**Assessment Category:** {cat.title()}")
                    st.write(f"**Model Internal Score:** {f'{score:.2%}' if score is not None else 'N/A'}")
                    st.write(f"**Model Architecture:** {pred.model_type}")
                    st.write(f"**Model Version:** {pred.model_version}")

                with mc2:
                    st.markdown("**SHAP Feature Contributions (Acoustic Biomarkers):**")
                    if pred.shap_values and pred.feature_names:
                        contributions = []
                        for i, fname in enumerate(pred.feature_names):
                            val = pred.shap_values[i] if i < len(pred.shap_values) else 0.0
                            contributions.append({
                                "feature": fname,
                                "description": fname,
                                "shap_value": val,
                                "direction": "elevates score" if val > 0 else "reduces score",
                            })
                        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                        fig = shap_waterfall_chart(contributions)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Detailed SHAP values not recorded for this prediction.")
            else:
                st.info("Clinical Notice: No voice acoustic features were supplied with this screening session. In accordance with clinical validation standards, no ML prediction was generated for symptom-only input.")

            # Clinician Notes Section
            st.markdown("---")
            st.markdown("#### Clinician Observations & Documentation")
            st.caption("Notes recorded here are stored securely and separate from automated AI outputs.")

            existing_notes = session.query(ClinicianNote).filter(
                ClinicianNote.screening_id == screening.id
            ).order_by(ClinicianNote.created_at.desc()).all()

            if existing_notes:
                for note in existing_notes:
                    with st.chat_message("user"):
                        st.markdown(f"**{note.doctor.full_name if note.doctor else 'Clinician'}** *({note.created_at.strftime('%Y-%m-%d %H:%M')})*")
                        st.markdown(note.note_text)
                        st.caption(f"Classification: {note.note_type.title()}")

            new_note = st.text_area("Add clinical observation / follow-up plan:", key=f"note_input_{screening.id}")
            note_type = st.selectbox("Note Type:", options=["observation", "recommendation", "follow_up"], key=f"ntype_{screening.id}")

            if st.button("Save Clinical Note", icon=":material/save:", type="primary", key=f"btn_save_note_{screening.id}"):
                if new_note.strip():
                    doc_record = session.query(User).filter(User.id == current_user.get("id")).first()
                    note_obj = ClinicianNote(
                        screening_id=screening.id,
                        doctor_id=doc_record.id if doc_record else current_user.get("id"),
                        note_text=new_note.strip(),
                        note_type=note_type,
                    )
                    session.add(note_obj)
                    session.commit()
                    st.success("Clinical note saved successfully.")
                    st.rerun()

            # Export Clinical PDF Summary
            st.markdown("---")
            try:
                report_id = f"CLINICAL-{screening.session_id[:8].upper()}"
                pdf_bytes = generate_screening_report(
                    report_id=report_id,
                    screening_data={
                        "age_range": screening.age_range,
                        "symptom_duration": screening.symptom_duration,
                        "existing_diagnosis": screening.existing_diagnosis,
                        "family_history": screening.family_history,
                        "natural_language_input": screening.natural_language_input,
                        "questionnaire_responses": screening.questionnaire_responses,
                    },
                    prediction_data={
                        "success": True,
                        "prediction_label": pred.prediction_label if pred else 0,
                        "prediction_category": pred.prediction_category if pred else "No voice data",
                        "prediction_score": pred.prediction_score if pred else None,
                        "model_version": pred.model_version if pred else "N/A",
                        "model_type": pred.model_type if pred else "N/A",
                    } if pred else None,
                )
                st.download_button(
                    "Download Clinical Summary (PDF)",
                    icon=":material/download:",
                    data=bytes(pdf_bytes),
                    file_name=f"clinical_summary_{report_id}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                logger.error(f"PDF generation error: {e}")

    except Exception as e:
        logger.error(f"Error inspecting screening {selected_db_id}: {e}")
        st.error(f"Error loading record details: {e}")
    finally:
        session.close()
