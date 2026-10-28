# PD-Aware — AI-Assisted Parkinson’s Disease Screening

> **Important Medical Disclaimer:** PD-Aware provides AI-assisted screening information for decision-support and educational purposes only. It is **not** a diagnostic device and does **not** diagnose or rule out Parkinson’s disease. It does not replace a clinical examination by a qualified neurologist or physician.

---

## 1. Overview & Vision

**PD-Aware** is an end-to-end, privacy-first web application designed for patients, caregivers, and medical professionals to screen for potential indicators associated with Parkinson’s disease (PD).

Rather than presenting mock statistics or speculative AI claims, PD-Aware enforces strict **clinical honesty and scientific integrity**:
- **Structured Screening Workflow:** Guides users through informed consent, demographics, natural-language symptom reporting (with reviewable NLP extraction), and multi-domain symptom questionnaires.
- **Biomedical Acoustic Analysis:** Evaluates 22 validated micro-acoustic features (vocal pitch, jitter, shimmer, harmonicity, recurrence period density entropy) trained on the UCI Oxford Parkinson's dataset.
- **Multimodal Separation:** Transparently distinguishes between questionnaire information and acoustic models. If voice data is omitted, no fabricated machine-learning scores are invented; a structured clinical summary is produced instead.
- **SHAP Model Explainability:** Local feature attributions (SHapley Additive exPlanations) reveal which acoustic properties influenced the model assessment without claiming medical causation.
- **Doctor Portal:** Authenticated clinical environment featuring patient record filtering, audit logging, and separate clinician notes storage.
- **Model Engineering Lab:** Interactive training pipeline supporting custom CSV upload, column mapping, automated dataset-quality checks, and multi-model benchmarking.

---

## 2. Leakage-Free Scientific Architecture

A critical vulnerability in biomedical machine learning is **data leakage**. PD-Aware implements explicit safeguards:

1. **Zero Subject-Level Leakage:**
   In datasets with multiple recordings per individual (such as the UCI dataset, containing 195 recordings from 31 subjects), standard k-fold or random splitting causes the model to memorize subject voice identities rather than disease patterns. PD-Aware uses `GroupShuffleSplit` by subject identifier (`name` column). Recordings from any given individual are placed **strictly** in either the training set or the test set—never split across both.
2. **Zero Preprocessing Leakage:**
   The `StandardScaler` feature normalization is fit **strictly on the training split**. Test data and new inference requests are transformed using training distribution parameters only.

---

## 3. Technology Stack

- **Application & UI:** Streamlit 1.45+, Plotly interactive charts
- **Machine Learning:** Scikit-Learn, XGBoost, SHAP (SHapley Additive exPlanations)
- **Natural Language Processing:** Regex-based symptom and temporal extraction with context extraction
- **Database & Persistence:** SQLAlchemy 2.0 ORM with SQLite (supports PostgreSQL via `DATABASE_URL`)
- **Authentication & Security:** Bcrypt salted password hashing, role-based access control (Patient, Doctor, Admin), and immutable action audit logging
- **Clinical Reports:** `fpdf2` generating non-diagnostic PDF screening summaries without fabricated credentials

---

## 4. Project Structure

```
pd-aware/
├── app.py                         # Application entrypoint & landing page
├── config/
│   └── settings.py                # System settings, feature definitions, disclaimers
├── data/
│   └── parkinsons.data            # Reference UCI Oxford Parkinson's dataset
├── database/
│   ├── engine.py                  # SQLAlchemy engine & session management
│   └── models.py                  # ORM models (User, Screening, Prediction, ClinicianNote, AuditLog)
├── ml/
│   ├── dataset.py                 # Subject-aware splitting, leakage checks, preprocessing
│   ├── train.py                   # Multi-algorithm training pipeline & artifact persistence
│   ├── evaluate.py                # Metrics evaluation (Acc, Prec, Rec, F1, ROC-AUC, PR-AUC, CM)
│   ├── inference.py               # Production inference service & schema validator
│   ├── explainability.py          # Local SHAP explanations & global feature importances
│   └── artifacts/                 # Versioned model artifacts (model, scaler, metadata.json)
├── nlp/
│   ├── symptom_extractor.py       # Rule-based NLP symptom and time extractor
│   └── rag_engine.py              # Clinical knowledge base for Parkinson's disease
├── components/
│   ├── auth_manager.py            # Bcrypt authentication, registration, session state
│   ├── charts.py                  # Plotly visualizations (SHAP waterfall, risk gauge, CM)
│   └── report_generator.py        # PDF clinical summary report generator
├── pages/
│   ├── 02_Start_Screening.py      # Guided multi-step patient screening workflow
│   ├── 03_How_It_Works.py         # Technical and clinical methodology explanation
│   ├── 04_Understanding_Parkinsons.py # Educational guide & clinical Q&A
│   ├── 05_Doctor_Dashboard.py     # Authenticated clinician portal & notes management
│   └── 06_Model_Lab.py            # CSV upload, dataset checks, training & evaluation suite
└── tests/
    ├── test_data_leakage.py       # Verification of zero subject-level & preprocessing leakage
    ├── test_dataset_quality.py    # Checks missing values, label consistency, dtypes
    ├── test_models_and_inference.py # Tests all candidate models, schemas, and SHAP
    ├── test_nlp_and_questionnaire.py# Tests keyword NLP extraction and question schemas
    ├── test_auth_and_security.py  # Tests bcrypt, session handling, roles, and audit logs
    ├── test_report_generation.py  # Tests PDF report creation and unicode sanitization
    └── test_end_to_end.py         # Full end-to-end integration test
```

---

## 5. Quickstart & Reproduction Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Conda or standard `venv`

### Step 1: Install Dependencies
```bash
cd "pd-aware"
pip install -r requirements.txt
```

### Step 2: Run the Automated Test Suite
To verify that all models, data-leakage guards, NLP extractors, and report generators are functioning with zero errors:
```bash
pytest -v
```
*Expected result: 28 passed, 0 failures.*

### Step 3: Train & Save Initial Model Artifacts
You can train candidate models and save the best-performing model (`v1.0.0`) via the CLI or via the web Model Lab:
```bash
python -m ml.train
```

### Step 4: Launch the Web Application
```bash
streamlit run app.py
```
Open your browser to the local URL (typically `http://localhost:8501`).

---

## 6. User Workflows

### A. General Public & Patients (`Start Screening`)
1. **Consent & Purpose:** Review screening scope and provide informed consent.
2. **Demographics:** Optional context (age group, symptom duration, family history).
3. **Natural-Language Description:** Write symptoms in plain text; the NLP engine extracts candidate items for user review and correction.
4. **Structured Questionnaire:** Rate severity of motor and non-motor symptoms.
5. **Optional Voice Biomarker Analysis:**
   - Choose to skip voice analysis (generates symptom-only clinical summary),
   - Load a verified reference case from the UCI dataset,
   - Upload custom acoustic features CSV, or
   - Enter manual acoustic measurements.
6. **Review & Database Submission:** Confirm all reported details.
7. **Results & PDF Report:** View model risk assessment with SHAP feature explanations (if voice data provided) or clinical symptom summary (if skipped), and download the PDF report.

### B. Clinicians (`Doctor Dashboard`)
- Authenticate with clinician credentials (or use the one-click demo access for Dr. Sharma, MD).
- Filter and search through patient screening sessions.
- Inspect patient narratives, questionnaire responses, and SHAP acoustic explanations.
- Add and persist separate clinician observations and recommendations.
- Download a formal clinical PDF report for the patient file.

### C. Researchers & Engineers (`Model Lab`)
- Ingest either the reference UCI dataset or upload custom CSV datasets.
- Configure column mappings for Subject ID, Target, and Features.
- Run automated dataset audits (missingness, class balance, subject consistency).
- Run subject-aware train/test splits with verified 0% subject overlap.
- Benchmark Logistic Regression, SVM, Random Forest, Extra Trees, Gradient Boosting, and XGBoost with full metrics (ROC-AUC, PR-AUC, Sensitivity, Specificity, F1).
- Export versioned artifacts and benchmark reports.

---

## 7. Known Limitations & Research Boundaries

1. **Dataset Size:** The primary voice model is trained on 195 recordings from 31 subjects. While subject-aware splitting prevents internal leakage, external cross-dataset validation is necessary to confirm broad clinical utility.
2. **Acoustic Focus:** Current ML models assess acoustic voice measurements, not questionnaire text.
3. **Screening vs. Diagnosis:** A high model score indicates that voice measurements share statistical similarities with Parkinson's training examples; it does not confirm a diagnosis. Comprehensive neurological evaluation is always required.

