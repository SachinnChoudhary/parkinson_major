Do not stop at the frontend or generate mock functionality. Inspect the current repository and implement the actual end-to-end workflow.

Start with CSV upload, column mapping, dataset-quality checks, subject-aware train/test splitting, training Logistic Regression/SVM/Random Forest, computing evaluation metrics, persisting experiments, and exporting a report.

Connect every interface element to a real backend operation. Remove fabricated dashboard statistics and placeholder results. Add automated tests for subject-level data leakage and preprocessing leakage.

Run the frontend and backend, execute the tests, fix failures, and document the exact commands required to reproduce the working application. Clearly list any features that remain incomplete.

# Project: PD-A ware — AI-Assisted Parkinson’s Disease Screening

## 1. Product Vision

Build a complete, production-oriented web application called **PD-Aware** that allows members of the public, patients, caregivers, and doctors to assess potential indicators associated with Parkinson’s disease using machine learning.

This is a public-facing health application, not a research dashboard.

The application should allow users to:
- Describe their symptoms in natural language.
- Select symptoms from a structured questionnaire.
- Enter relevant medical and demographic information where appropriate.
- Upload supported test data, such as voice recordings, if the corresponding model is implemented and validated.
- Receive an understandable, model-generated risk estimate.
- Understand which entered features influenced the prediction.
- Receive appropriate next-step guidance.
- Save, download, or share a screening summary with a doctor.
- Allow doctors to review patient-submitted information and screening summaries.

The interface must work for both nontechnical users and medical professionals.

The underlying application should also retain sufficient technical information about models, validation methods, datasets, and predictions to support a scientific evaluation and a future research paper.

**Important:** The application must not claim to diagnose Parkinson’s disease. A machine-learning prediction is not a confirmed diagnosis and must not replace an examination by a qualified clinician.

---

## 2. Target Users

### A. General Public and Patients
Provide a simple, guided screening experience without technical terminology.

Users should be able to:
1. Start a screening session.
2. Describe their concerns in ordinary language.
3. Answer structured symptom questions.
4. Review their answers.
5. Submit the information for analysis.
6. Receive a clearly explained screening result.
7. Download a summary to discuss with a doctor.

Do not require users to understand machine learning, biomarkers, model metrics, or statistical terminology.

### B. Doctors and Healthcare Professionals
Provide a professional workflow that allows doctors to:
- Review a patient's reported symptoms.
- Examine available patient-submitted screening information.
- Review the model's output and its limitations.
- Inspect the factors contributing to a prediction.
- Generate or download a screening summary.
- Document their own clinical observations separately from the AI output.

Do not represent the application as a replacement for neurological examination or established clinical diagnostic procedures.

### C. Application Administrator
Provide a secure administrative area for managing:
- Model versions.
- Published model configurations.
- Application settings.
- Dataset and model metadata.
- System health and error logs.
- Access permissions.

The administrator must not be able to present unvalidated models as clinically approved merely by changing a setting.

---

## 3. Recommended Technology Stack

### Frontend
- Next.js with App Router.
- TypeScript.
- Tailwind CSS.
- shadcn/ui.
- Lucide icons.
- React Hook Form.
- Zod validation.
- Recharts for appropriate visualizations.

### Backend
- Python.
- FastAPI.
- Pydantic.
- scikit-learn.
- XGBoost, where appropriate.
- SHAP or permutation importance for supported explanations.
- pandas and NumPy.
- SciPy and librosa for optional audio analysis.

### Data and Infrastructure
- PostgreSQL for production application data.
- SQLAlchemy and database migrations.
- Secure object storage for uploaded files, if required.
- Docker for reproducible deployment.
- Environment variables for secrets.
- Automated tests and structured logging.

Keep the frontend, backend, inference pipeline, and database logic modular.

Do not put model inference or sensitive business logic entirely in the browser.

---

## 4. User Interface and Experience

Create a modern, accessible, trustworthy healthcare interface.

Design principles:
- Clean, minimal, professional layout.
- Light background with restrained blue, teal, and neutral accents.
- Readable typography and sufficient spacing.
- Responsive desktop, tablet, and mobile layouts.
- Clear forms and progress indicators.
- Accessible labels and keyboard navigation.
- Plain-language explanations.
- No unnecessary medical animations or alarming red warning screens.
- No exaggerated claims, fake testimonials, or fabricated clinical statistics.

### Main Navigation

Public navigation:
- Home
- Start Screening
- How It Works
- Understanding Parkinson’s
- For Doctors
- Privacy and Safety
- About

Authenticated navigation, where applicable:
- My Screening History
- My Reports
- Profile and Consent
- Doctor Dashboard, for authorized professional accounts

Keep internal model diagnostics and engineering tools out of the normal patient-facing navigation.

---

## 5. Home Page

Create a professional landing page explaining:
- What Parkinson’s disease is.
- What the application can and cannot do.
- How a screening session works.
- Which types of information the application supports.
- Why a model-generated result should be discussed with a healthcare professional.

Primary action: **Start Screening**

Secondary action: **For Healthcare Professionals**

Display a visible disclaimer:

“NeuroAI provides AI-assisted screening information for informational and decision-support purposes. It does not diagnose or rule out Parkinson’s disease. Consult a qualified healthcare professional for clinical assessment.”

Do not claim that NeuroAI has been clinically validated unless appropriate evidence is available.

---

## 6. Patient Screening Workflow

Build a guided, multi-step screening process.

### Step 1: Introduction and Consent

Explain:
- The purpose of the screening.
- What information will be collected.
- How the information will be used.
- Whether information will be stored.
- Whether any external service will receive information.
- The limitations of AI-generated predictions.

Request informed consent before collecting or processing sensitive health information.

Allow users to understand the privacy implications before continuing.

### Step 2: Patient-Reported Information

Collect only information necessary for the implemented screening workflow.

Potential fields include:
- Age range or age, where justified.
- Symptom duration.
- Relevant medical history.
- Current medications, if relevant to the model or clinical summary.
- Existing Parkinson’s diagnosis, where relevant to the selected workflow.
- Family history, if supported by the model or questionnaire.

Make sensitive fields optional when they are not necessary.

Do not infer a patient's medical history from unrelated information.

### Step 3: Natural-Language Symptom Description

Provide a text box where users can explain their concerns in their own words.

Example:
“I have noticed shaking in my right hand when it is resting. My handwriting has become smaller, and I feel slower when walking.”

Use an NLP component to extract candidate symptoms and relevant time-related details.

Requirements:
- Show extracted information to the user before using it.
- Let the user correct, remove, or add symptoms.
- Preserve uncertainty when the description is ambiguous.
- Do not interpret every symptom as evidence of Parkinson’s disease.
- Do not invent symptoms the user did not report.
- Do not allow an LLM to independently determine the final model prediction.
- Use deterministic, validated mappings between extracted symptoms and model features.

If a local LLM is available through Ollama, it may assist with language understanding and plain-language explanations. It must not be treated as a clinically validated diagnostic model.

### Step 4: Structured Symptom Questionnaire

Organize questions into clear categories.

Potential categories include:

**Movement-related symptoms**
- Tremor at rest.
- Slowness of movement.
- Muscle stiffness.
- Changes in walking.
- Reduced arm swing.
- Balance difficulties.
- Changes in handwriting.
- Reduced facial expression.
- Softer speech.

**Other reported symptoms**
- Reduced sense of smell.
- Constipation.
- Sleep-related changes.
- Changes in mood.
- Fatigue.
- Other concerns.

These are candidate questionnaire items, not a claim that each symptom is specific to Parkinson’s disease.

For every question:
- Provide clear answer options.
- Include “Not sure” where appropriate.
- Distinguish absence of a symptom from missing information.
- Avoid suggesting that one symptom establishes a diagnosis.
- Include an option to describe other symptoms.

Do not assume that all these questions can be passed directly into a trained model. Only features supported by the selected model may be used for inference.

### Step 5: Optional Supported Tests

Design the architecture to support additional screening inputs, but activate a test type only when a compatible model and validated processing pipeline are available.

Potential future modules:
- Voice feature analysis.
- Speech timing analysis.
- Handwriting or spiral-drawing analysis.
- Gait-related measurements.
- Clinician-entered assessment scores.

For voice analysis, potential features include:
- Jitter.
- Shimmer.
- Harmonics-to-noise ratio.
- MFCCs.
- Supported speech timing and spectral features.

Clearly distinguish research-grade voice analysis from clinically established assessments.

Do not pretend that a voice recording, drawing, or questionnaire can independently establish Parkinson’s disease.

### Step 6: Review and Submission

Show a review page summarizing:
- User-entered information.
- Selected symptoms.
- Missing or uncertain answers.
- Optional test data.
- Consent status.

Let the user correct information before submission.

### Step 7: Model Inference

Send supported features to the backend for inference.

The backend must:
1. Validate the request.
2. Check feature compatibility.
3. Apply the preprocessing pipeline associated with the selected model version.
4. Generate the model output.
5. Calculate an uncertainty or confidence measure only where technically justified.
6. Record model version and inference metadata.
7. Return a structured result.

Never fabricate a prediction when the model is unavailable, the request is invalid, or required inputs are missing.

If the user's inputs are incompatible with the model, explain what is missing and allow the user to correct the information.

---

## 7. Model Development and Deployment

Build a real machine-learning inference service.

Do not create a mock prediction system and present it as functional AI.

### Initial Model

Start with a properly evaluated tabular machine-learning model trained on an appropriate, documented dataset.

Candidate algorithms:
- Logistic Regression.
- Support Vector Machine.
- Random Forest.
- Extra Trees.
- Gradient Boosting.
- XGBoost, if appropriate.

Select the final model based on valid evaluation, not merely the highest accuracy.

### Critical Feature Compatibility Requirement

The model must only receive features that match its training schema.

A model trained on acoustic voice measurements cannot directly predict from a patient's natural-language symptom description.

Likewise, a model trained on a limited set of questionnaire features cannot accept arbitrary symptoms unless a justified, tested feature-mapping pipeline exists.

If the available dataset does not contain suitable symptom questionnaire data, do not manufacture training labels or assume that voice-model performance applies to symptoms.

Instead:
- Implement the available model as its own clearly labelled screening module.
- Build the symptom questionnaire as a separate information-gathering workflow.
- Keep unsupported prediction paths disabled.
- Design a future multimodal architecture that can be activated once appropriate data and validation are available.

### Model Artifact Management

Store model artifacts with metadata:
- Model identifier.
- Version.
- Training dataset provenance.
- Supported features.
- Preprocessing version.
- Validation protocol.
- Evaluation metrics.
- Training date.
- Intended use.
- Known limitations.
- Approval or release status.

Load model artifacts securely.

Do not load arbitrary uploaded model files in the public inference service.

---

## 8. Results Page

The results page is the most important patient-facing component.

Use calm, clear, non-alarming language.

### Result Structure

Include:
1. Screening summary.
2. Model-generated output, described accurately.
3. A plain-language explanation of what the output means.
4. Factors that influenced the model's prediction, when supported.
5. Important limitations.
6. Recommended next steps.
7. A downloadable report.

### Risk Language

Do not invent risk percentages or convert a raw classifier score into a clinical probability without appropriate calibration and validation.

If a validated model supports a calibrated probability, explain what that probability represents and its limitations.

Otherwise, display a clearly labelled model score or category, with an explanation of why it must not be interpreted as an individual's clinical probability.

Do not display an arbitrary “Low / Medium / High risk” classification unless its thresholds have been justified and evaluated for the intended use.

Do not say:
- “You have Parkinson’s disease.”
- “You definitely do not have Parkinson’s disease.”
- “Parkinson’s detected with 99% certainty.”
- “You are completely healthy.”

Prefer language such as:

“The model identified patterns in the information you provided that were more consistent with its Parkinson’s-associated training examples. This result is not a diagnosis and cannot establish whether you have Parkinson’s disease.”

The wording must accurately reflect the actual model, output, and validation evidence.

### Next-Step Guidance

Provide general, non-diagnostic guidance to discuss persistent or concerning symptoms with a qualified healthcare professional.

For symptoms that could indicate an emergency, use a separately reviewed safety pathway rather than attributing them to Parkinson’s disease.

Do not automatically recommend medication, change medication doses, or prescribe treatment.

---

## 9. Explainability

Provide explanations that help users understand the model's output without overstating what the model knows.

For supported models, use:
- SHAP explanations.
- Permutation importance.
- Feature contribution summaries.
- Local explanations for an individual prediction.

Show:
- Which input features influenced the output.
- Whether their influence increased or decreased the model score, where technically supported.
- A plain-language description of each feature.
- Important limitations of the explanation.

Do not claim that feature importance proves causation.

Do not tell a patient that a particular symptom is the medical cause of their condition merely because the model assigned it high importance.

Patient-facing explanations should avoid unexplained statistical terminology. Offer technical detail in the professional interface where appropriate.

---

## 10. Patient Reports

Allow users to download a PDF screening summary.

The report should contain:
- Screening date.
- Unique report identifier.
- User-entered information that is appropriate to include.
- Reported symptoms.
- Supported test results.
- Model output and its correct interpretation.
- Relevant explanation of contributing features.
- Model version.
- Limitations.
- Suggested clinical follow-up.
- A prominent statement that the report is not a medical diagnosis.

Do not include fabricated clinician names, hospital affiliations, signatures, certifications, or approval seals.

Clearly label AI-generated information and distinguish it from clinician-entered observations.

Allow users to choose what information is included before sharing the report.

---

## 11. Doctor Dashboard

Build a separate, authenticated interface for healthcare professionals.

Features:
- Patient screening summaries, where access has been authorized.
- Search and filtering.
- Patient-reported symptoms.
- Supported model outputs.
- Feature-level explanations.
- Downloadable reports.
- Clinician notes, stored separately from AI output.
- Model limitations and version information.

Security requirements:
- Authentication.
- Role-based authorization.
- Access logging.
- Secure session handling.
- Patient consent and appropriate access controls.
- Protection against cross-user data exposure.

Do not assume that an account belongs to a licensed doctor simply because the user selects “Doctor.” If the application makes professional credentials a condition of access, implement an appropriate verification process.

Do not claim compliance with any healthcare regulation unless it has actually been assessed and implemented.

---

## 12. Privacy and Security

Treat health information as sensitive.

Implement:
- Data minimization.
- HTTPS in deployment.
- Secure password handling.
- Role-based access control.
- Input validation.
- Upload type and size limits.
- Secure file storage.
- Protection against malicious file uploads and ZIP path traversal.
- Rate limiting.
- Audit logs.
- Configurable retention and deletion.
- Explicit consent for storage and processing.
- Safe error messages that do not expose sensitive information.
- Secrets stored outside source code.

Do not send patient information to third-party AI services by default.

If an external service is used, disclose it and obtain appropriate consent.

If local inference is available, provide a local-processing configuration where feasible.

Never log full patient narratives, raw audio, or sensitive medical details unnecessarily.

---

## 13. Research and Quality Tracking Behind the Scenes

The product is for the public and doctors, but the implementation must retain sufficient metadata to support a future research paper.

This should be an internal quality and audit capability, not the primary user interface.

Record, where appropriate:
- Model version.
- Feature schema version.
- Preprocessing version.
- Inference timestamp.
- Validation status of the deployed model.
- Non-identifying technical error information.
- User consent for any permitted use of data.

Do not automatically use patient information for research or model training.

Any later research use must have a justified legal and ethical basis, appropriate consent or other applicable authorization, and institutional review where required.

Do not claim that a deployed model is clinically effective based only on retrospective accuracy on a public dataset.

The research paper should later report the actual methods, limitations, validation results, and observed outcomes. The application must not manufacture evidence to make the paper appear stronger.

---

## 14. Testing Requirements

Create automated tests for:
- Questionnaire validation.
- Natural-language extraction and user correction.
- Feature-schema compatibility.
- Model inference.
- Missing-data handling.
- Invalid inputs.
- Model version loading.
- Prediction response structure.
- Report generation.
- Authentication and authorization.
- Patient-data isolation.
- File upload security.
- Consent and deletion workflows.
- Failure handling when a model is unavailable.

Test the complete journey from screening start to downloadable report.

Use test fixtures and synthetic patient records for development. Do not use fabricated clinical performance results in the live application.

---

## 15. Implementation Strategy

Build the application in stages.

### Phase 1: Functional MVP
- Landing page.
- Patient screening workflow.
- Structured symptom questionnaire.
- Natural-language input with reviewable extraction.
- One real model, but only if a suitable trained model and matching input schema are available.
- Honest result page.
- PDF report.
- Basic privacy and security controls.

If no appropriate model is ready, implement the screening workflow and clearly state that prediction is not yet available. Do not create fake AI outputs.

### Phase 2: Doctor Experience
- Authentication.
- Doctor dashboard.
- Authorized report access.
- Clinician notes.
- Improved PDF reports.

### Phase 3: Additional Screening Modalities
- Voice analysis, if supported by a properly evaluated model.
- Other supported test modalities.
- Model versioning.
- Explainability.
- Calibration and uncertainty handling.

### Phase 4: Production Hardening
- Security review.
- Performance testing.
- Accessibility review.
- Error monitoring.
- Backup and recovery.
- Deployment documentation.
- Model release and rollback procedures.

### Phase 5: Scientific Evaluation
- Evaluate generalization to unseen people.
- Use subject-level validation where appropriate.
- Perform external validation when compatible independent datasets are available.
- Report sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, and calibration where applicable.
- Quantify uncertainty in evaluation results.
- Investigate dataset bias and limitations.
- Document methods sufficiently for reproducibility.

These evaluations support a later research paper; they do not automatically establish clinical utility.

---

## 16. Final Acceptance Criteria

The project is complete only when:

1. The application has a polished, public-facing interface.
2. Users can complete the screening workflow.
3. Natural-language symptom extraction is reviewable and correctable.
4. Model predictions use only supported, compatible features.
5. No fake predictions or invented clinical metrics are displayed.
6. Results are understandable and appropriately qualified.
7. Reports can be downloaded.
8. Doctor features enforce appropriate access controls.
9. Sensitive data is handled securely.
10. Automated tests cover critical workflows.
11. The README explains setup, configuration, model limitations, and deployment.
12. The application clearly distinguishes screening information from medical diagnosis.
13. The codebase supports future model evaluation and scientific reporting.

Build a real, working application rather than a static UI mockup. Start by inspecting the existing repository, then implement the frontend, backend, database, inference pipeline, and tests in logical stages. Document every unavailable or unvalidated capability instead of pretending it works.