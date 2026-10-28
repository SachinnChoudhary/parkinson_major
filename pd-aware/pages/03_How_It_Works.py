"""How It Works page — explains the screening methodology."""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import APP_NAME

st.set_page_config(page_title=f"How It Works — {APP_NAME}", layout="wide")

st.markdown("## How PD-Aware Works")

st.markdown("""
PD-Aware is an AI-assisted screening tool that uses machine learning to help identify
potential indicators associated with Parkinson's disease. Here is a detailed explanation
of each step in the process.
""")

# Step-by-step explanation
st.markdown("### 1. Data Collection")
st.markdown("""
We collect information through two complementary methods:

- **Natural Language Description:** You describe your symptoms in your own words.
  Our NLP system identifies potential symptoms, which you can review and correct.
- **Structured Questionnaire:** You answer specific questions about motor and
  non-motor symptoms, selecting the severity level for each.
- **Voice Data (if available):** The model can analyze acoustic voice features
  when voice recordings are provided.

All information is collected with your explicit consent and stored locally.
""")

st.markdown("### 2. The Machine Learning Model")
st.markdown("""
Our current model is trained on the **UCI Oxford Parkinson's Disease Detection Dataset**:

| Property | Details |
|----------|---------|
| **Dataset** | 195 voice recordings from 31 subjects |
| **Subjects** | 23 with Parkinson's, 8 healthy controls |
| **Features** | 22 voice biomedical measurements |
| **Key Features** | Jitter, Shimmer, HNR, RPDE, DFA, PPE |
| **Splitting** | Subject-aware (no data leakage) |
| **Algorithms** | Logistic Regression, SVM, Random Forest, XGBoost, etc. |
| **Selection** | Composite score: 0.4×ROC-AUC + 0.3×F1 + 0.3×Sensitivity |

**Important:** This model analyzes voice features, not symptom questionnaire responses.
The symptom questionnaire collects information for your screening summary and future
model development, but the current voice model requires acoustic measurements.
""")

st.markdown("### 3. Preventing Data Leakage")
st.markdown("""
Data leakage is a critical problem in machine learning where information from the
test set "leaks" into the training process, giving artificially high performance.

Our approach prevents two types of leakage:

- **Subject-level leakage:** All recordings from a single person go entirely into
  either the training or test set — never both. Without this, the model would
  effectively memorize individuals rather than learning general patterns.
- **Preprocessing leakage:** The feature scaler is fit only on the training data.
  Test data is scaled using the training statistics, never its own.
""")

st.markdown("### 4. Model Explanations")
st.markdown("""
We use **SHAP (SHapley Additive exPlanations)** to explain individual predictions:

- Each feature gets a SHAP value showing how it pushed the prediction
- Positive SHAP values push toward a Parkinson's-associated prediction
- Negative values push toward a healthy prediction
- The magnitude shows how strongly each feature influenced the result

**Important:** Feature importance shows correlation, not causation. A high-importance
feature doesn't mean it's the medical cause of a condition.
""")

st.markdown("### 5. Limitations")
st.markdown("""
- **Small dataset:** Only 31 subjects — results may not generalize broadly.
- **Voice features only:** Current model uses acoustic measurements, not clinical symptoms.
- **Not clinically validated:** This is a research tool, not a diagnostic device.
- **Population bias:** The training data may not represent all demographics.
- **No substitute for clinical evaluation:** Always consult a healthcare professional.
""")

st.markdown("### 6. Privacy")
st.markdown("""
- All processing happens locally on this machine
- No data is sent to external services
- You control what information is stored
- You can delete your data at any time
- Screening results are stored in a local SQLite database
""")
