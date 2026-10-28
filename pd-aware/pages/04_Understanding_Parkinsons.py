"""
Understanding Parkinson's Disease — Educational resource and knowledge base.

Provides accessible, medically grounded information for patients, caregivers,
and clinicians regarding Parkinson's disease, motor and non-motor symptoms,
voice biomarkers, and when to seek medical evaluation.
"""

import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import APP_NAME, MEDICAL_DISCLAIMER
from nlp.rag_engine import PARKINSONS_KNOWLEDGE

st.set_page_config(
    page_title=f"Understanding Parkinson's — {APP_NAME}",
    layout="wide",
)

st.markdown("## Understanding Parkinson's Disease")
st.markdown("""
Parkinson's disease (PD) is a progressive neurodegenerative disorder affecting movement coordination
and non-motor autonomic functions. This clinical guide outlines the underlying pathophysiology,
cardinal indicators, voice biomarkers, and diagnostic approaches.
""")

st.markdown(f"""
<div style="background: #FEF9C3; border: 1px solid #FCD34D;
border-radius: 10px; padding: 1rem 1.5rem; margin: 1rem 0; font-size: 0.85rem;
color: #92400E;">
    <strong>Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)

# ─── Navigation Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Causes",
    "Motor & Non-Motor Symptoms",
    "Voice Biomarkers Explained",
    "Diagnosis & Clinical Care",
    "Educational Knowledge Base",
])

# ─── TAB 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Pathophysiology & Neurological Basis")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        Parkinson's disease is a progressive movement disorder characterized by the gradual degeneration
        of dopaminergic neurons in the brainstem.

        **Key Pathological Mechanisms:**
        - **Dopamine Depletion:** Nerve cells in the **substantia nigra pars compacta** normally produce
          **dopamine**, a neurotransmitter essential for smooth, purposeful motor control.
        - **Neuronal Loss:** In Parkinson's disease, these neurons undergo apoptosis. By the time primary
          motor manifestations appear, 60–80% of substantia nigra dopaminergic neurons may already be compromised.
        - **Lewy Pathology:** Intracellular protein aggregates called **Lewy bodies**—principally composed of
          misfolded alpha-synuclein—develop within surviving neurons and propagate throughout neural circuits.
        """)
    with col2:
        st.info("""
        **Epidemiology & Prognosis:**
        - **Onset:** Typically manifests after age 60, though early-onset cases occur prior to age 50.
        - **Prevalence:** Affects approximately 1% of the population over 60, rising to 2-3% past age 80.
        - **Heterogeneity:** Progression rate and dominant symptom presentation vary considerably across individuals.
        - **Management:** Although neuroprotective cures are under investigation, symptomatic dopamine replacement therapy and multidisciplinary care significantly improve functional quality of life.
        """)

# ─── TAB 2: Symptoms ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Cardinal Signs & Clinical Manifestations")

    st.markdown("#### Primary Motor Criteria (TRAP Framework)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        - **Tremor at Rest:**
          A 4–6 Hz rhythmic oscillation typically starting unilaterally in an upper extremity
          ("pill-rolling"). Typically suppresses during purposeful movement and sleep.
        - **Rigidity:**
          Velocity-independent resistance to passive range of motion ("cogwheel" or "lead-pipe"
          stiffness) affecting the limbs, neck, or trunk.
        """)
    with c2:
        st.markdown("""
        - **Akinesia / Bradykinesia:**
          Slowness in initiating and executing movements, with progressive reduction in amplitude
          or speed during repetitive activities (e.g., finger tapping, foot stomping).
        - **Postural Instability:**
          Impaired righting reflexes resulting in unsteadiness and increased fall risk in mid-to-advanced stages.
        """)

    st.markdown("---")
    st.markdown("#### Non-Motor (Prodromal) Indicators")
    st.markdown("""
    Non-motor features frequently appear years before motor onset, providing crucial clinical context:
    """)
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        st.markdown("""
        **Sensory & Sleep:**
        - **Olfactory Dysfunction:** Marked loss of smell (hyposmia) preceding motor symptoms by years.
        - **RBD (REM Sleep Behavior Disorder):** Loss of normal motor paralysis during REM sleep, leading to vigorous dream enactment.
        """)
    with nc2:
        st.markdown("""
        **Autonomic Dysfunction:**
        - **Gastrointestinal Dysmotility:** Chronic constipation not attributable to dietary changes.
        - **Orthostatic Hypotension:** Lightheadedness upon standing due to impaired sympathetic baroreflexes.
        """)
    with nc3:
        st.markdown("""
        **Neuropsychiatric & Systemic:**
        - **Apathy & Depressive Affect:** Decreased motivation, unexplained mood alterations.
        - **Executive Functioning Changes:** Slower mental processing speed and multitasking difficulty.
        """)

# ─── TAB 3: Voice Biomarkers ──────────────────────────────────────────────────
with tab3:
    st.markdown("### Hypokinetic Dysarthria & Acoustic Analysis")
    st.markdown("""
    Speech and voice alterations affect up to **90% of individuals with Parkinson's disease** (hypokinetic dysarthria).

    These changes stem from physiological rigidity, reduced vocal cord (laryngeal) adduction range,
    and impaired auditory self-perception (patients often do not perceive their volume as reduced).
    """)

    st.markdown("#### The 22 Micro-Acoustic Variables Measured")
    st.markdown("""
    The dataset utilized in PD-Aware evaluates 22 biomedical measurements extracted from sustained vowel phonation (/a/):
    """)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        **Frequency Perturbation (Jitter):**
        - **MDVP:Fo(Hz):** Mean vocal fundamental frequency (pitch).
        - **MDVP:Fhi & Flo:** Maximum and minimum fundamental frequencies.
        - **Jitter (%, Abs, RAP, PPQ, DDP):** Minute cycle-to-cycle variability in vibration timing. Tremor or laryngeal stiffness causes elevated frequency irregularity.

        **Amplitude Perturbation (Shimmer):**
        - **Shimmer (dB, APQ3, APQ5, APQ, DDA):** Cycle-to-cycle variability in phonation amplitude (loudness). Incomplete glottal closure produces elevated shimmer.
        """)
    with v2:
        st.markdown("""
        **Noise & Harmonicity Ratios:**
        - **HNR (Harmonics-to-Noise Ratio):** Ratio of periodic tonal energy to turbulent acoustic noise. Reduced HNR reflects breathiness and hoarseness.
        - **NHR (Noise-to-Harmonics Ratio):** Inverse proportion quantifying noise contamination.

        **Nonlinear & Fractal Dynamics:**
        - **RPDE (Recurrence Period Density Entropy):** Measures phase space chaos and vocal cord biomechanical unpredictability.
        - **DFA (Detrended Fluctuation Analysis):** Quantifies fractal scaling properties in speech signal decay.
        - **PPE (Pitch Period Entropy):** Quantifies impaired fundamental frequency control on a logarithmic scale.
        """)

# ─── TAB 4: Diagnosis ─────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Clinical Diagnostic Pathways")
    st.markdown("""
    **Important Clinical Principle:** There is currently **no standalone laboratory blood test or routine scan** that definitively confirms Parkinson's disease.

    Established clinical diagnosis relies upon:
    1. **Formal Clinical Evaluation:** Detailed history of symptom evolution, unilateral onset, and progression.
    2. **Neurological Examination:** Assessment of bradykinesia combined with either rest tremor, rigidity, or both, evaluated via the MDS-UPDRS (Movement Disorder Society Unified Parkinson's Disease Rating Scale).
    3. **Dopaminergic Response:** Meaningful clinical symptom relief following an empirical trial of levodopa strongly supports the diagnosis.
    4. **Imaging Support (DaTscan, MRI):** DaTscan (dopamine transporter SPECT imaging) assesses striatal dopamine uptake to distinguish PD from essential tremor, while MRI excludes structural lesions or normal pressure hydrocephalus.
    """)

# ─── TAB 5: Educational Knowledge Base ────────────────────────────────────────
with tab5:
    st.markdown("### Educational Knowledge Search")
    st.markdown(
        "Search our clinical reference library for evidence-based information regarding Parkinson's disease."
    )

    query = st.text_input(
        "Search topic or clinical question:",
        placeholder="e.g., What is bradykinesia? What are non-motor signs? How does voice change occur?",
    )

    if query:
        st.markdown("#### Clinical Reference Findings:")
        matches = []
        q_lower = query.lower()

        for item in PARKINSONS_KNOWLEDGE:
            topic = item.get("topic", "")
            content = item.get("content", "")
            score = 0
            for word in q_lower.split():
                if len(word) > 2:
                    if word in topic.lower():
                        score += 3
                    if word in content.lower():
                        score += 1
            if score > 0:
                matches.append((score, topic, content, item.get("source", "Medical reference")))

        matches.sort(key=lambda x: x[0], reverse=True)

        if matches:
            for _, topic, content, source in matches[:3]:
                st.markdown(f"**{topic}**")
                st.write(content)
                st.caption(f"Source: {source}")
                st.markdown("---")
        else:
            st.info("No direct topic match found in the local knowledge base. Please consult a licensed neurologist for individual inquiries.")
    else:
        st.markdown("**Sample Clinical Reference Topics:**")
        for item in PARKINSONS_KNOWLEDGE[:4]:
            with st.expander(f"Reference: {item['topic']}", expanded=False):
                st.write(item["content"])
                st.caption(f"Source: {item['source']}")
