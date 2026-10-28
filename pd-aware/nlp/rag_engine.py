"""
RAG (Retrieval-Augmented Generation) engine for Parkinson's disease knowledge.

Uses a local knowledge base of Parkinson's disease information to provide
accurate, referenced answers via Ollama LLM.

IMPORTANT: The RAG system provides educational information only.
It does not diagnose conditions or replace medical advice.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import KNOWLEDGE_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ─── Built-in Knowledge Base ─────────────────────────────────────────────────
# Core Parkinson's disease information for RAG context

PARKINSONS_KNOWLEDGE = [
    {
        "topic": "What is Parkinson's Disease",
        "content": (
            "Parkinson's disease is a progressive neurological condition that affects "
            "movement. It develops gradually, sometimes starting with a barely noticeable "
            "tremor in just one hand. While tremor is the most well-known sign, the "
            "condition also commonly causes stiffness or slowing of movement. "
            "Parkinson's disease is caused by a loss of nerve cells in the part of the "
            "brain called the substantia nigra, which leads to a reduction in dopamine. "
            "Dopamine plays a vital role in regulating body movement."
        ),
        "source": "General medical knowledge",
    },
    {
        "topic": "Common Motor Symptoms",
        "content": (
            "The main motor symptoms of Parkinson's disease include: "
            "1) Tremor - rhythmic shaking, usually beginning in the hand or fingers, "
            "often at rest. 2) Bradykinesia - slowness of movement, making simple "
            "tasks difficult and time-consuming. 3) Rigidity - muscle stiffness that "
            "can limit range of motion and cause pain. 4) Postural instability - "
            "impaired balance and coordination. Other motor symptoms include reduced "
            "facial expression, changes in handwriting (micrographia), softer speech, "
            "and changes in walking pattern."
        ),
        "source": "General medical knowledge",
    },
    {
        "topic": "Non-Motor Symptoms",
        "content": (
            "Parkinson's disease also involves non-motor symptoms that can sometimes "
            "precede motor symptoms by years. These include: loss of sense of smell "
            "(anosmia), sleep disturbances including REM sleep behavior disorder, "
            "constipation and other autonomic dysfunction, depression and anxiety, "
            "cognitive changes, fatigue, and pain. Non-motor symptoms can significantly "
            "impact quality of life."
        ),
        "source": "General medical knowledge",
    },
    {
        "topic": "Diagnosis",
        "content": (
            "There is no single definitive test for Parkinson's disease. Diagnosis is "
            "primarily clinical, based on medical history, signs, symptoms, and a "
            "neurological and physical examination. A neurologist may use specific "
            "criteria such as the UK Brain Bank criteria or the MDS clinical "
            "diagnostic criteria. Imaging tests like DaTscan may support the diagnosis. "
            "It is important to note that AI screening tools cannot diagnose "
            "Parkinson's disease and should not replace clinical evaluation."
        ),
        "source": "General medical knowledge",
    },
    {
        "topic": "Voice Changes in Parkinson's",
        "content": (
            "Voice and speech changes are common in Parkinson's disease, affecting "
            "up to 90% of people with the condition. These changes include reduced "
            "volume (hypophonia), monotone speech, hoarseness, and imprecise articulation. "
            "Research has shown that measurable voice features such as jitter (frequency "
            "variation), shimmer (amplitude variation), harmonics-to-noise ratio (HNR), "
            "and other acoustic measurements can differ between people with and without "
            "Parkinson's disease. These features form the basis of voice-based "
            "screening approaches, though their clinical utility requires further "
            "validation."
        ),
        "source": "Little et al., 2007; research literature",
    },
    {
        "topic": "Treatment",
        "content": (
            "While there is currently no cure for Parkinson's disease, treatments can "
            "help manage symptoms. Treatment options include: medication (such as "
            "levodopa, dopamine agonists, MAO-B inhibitors), physical therapy, "
            "occupational therapy, speech therapy, and in some cases, surgical "
            "procedures like deep brain stimulation (DBS). Treatment plans are "
            "individualized and managed by healthcare professionals."
        ),
        "source": "General medical knowledge",
    },
    {
        "topic": "When to See a Doctor",
        "content": (
            "You should consider seeing a healthcare professional if you experience: "
            "persistent tremor, especially at rest; noticeable slowness in movements; "
            "muscle stiffness that doesn't resolve; changes in walking or balance; "
            "significant changes in handwriting; persistent changes in voice; or "
            "multiple non-motor symptoms such as loss of smell, sleep disturbances, "
            "and constipation occurring together. Early consultation is recommended "
            "because early intervention can help manage symptoms more effectively."
        ),
        "source": "General medical guidance",
    },
    {
        "topic": "Risk Factors",
        "content": (
            "Known risk factors for Parkinson's disease include: increasing age "
            "(most common after 60), male sex (slightly higher risk), family history "
            "(having a close relative with the condition), and certain genetic "
            "mutations. Environmental factors such as exposure to certain pesticides "
            "and herbicides have been associated with increased risk in some studies. "
            "However, having risk factors does not mean a person will develop "
            "Parkinson's disease."
        ),
        "source": "Epidemiological research",
    },
]


class RAGEngine:
    """
    Simple RAG engine using built-in knowledge base and optional Ollama LLM.

    If Ollama is not available, returns relevant knowledge base entries
    directly without LLM summarization.
    """

    def __init__(self):
        self.knowledge_base = PARKINSONS_KNOWLEDGE
        self.ollama_available = None

    async def check_ollama(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                self.ollama_available = response.status_code == 200
                return self.ollama_available
        except Exception:
            self.ollama_available = False
            return False

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant knowledge base entries.

        Uses simple keyword matching. A more sophisticated version would
        use embeddings and vector search.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for entry in self.knowledge_base:
            # Score by keyword overlap
            content_words = set(entry["content"].lower().split())
            topic_words = set(entry["topic"].lower().split())

            content_overlap = len(query_words & content_words)
            topic_overlap = len(query_words & topic_words) * 3  # Weight topic matches

            score = content_overlap + topic_overlap
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    async def answer(self, question: str) -> Dict:
        """
        Answer a question using RAG approach.

        1. Retrieve relevant knowledge
        2. If Ollama available, generate a contextualized answer
        3. Otherwise, return retrieved context directly
        """
        # Retrieve relevant context
        context_entries = self.retrieve(question)

        if not context_entries:
            return {
                "answer": (
                    "I don't have specific information about that topic. "
                    "For medical questions, please consult a healthcare professional."
                ),
                "sources": [],
                "method": "no_match",
            }

        context_text = "\n\n".join([
            f"**{entry['topic']}**: {entry['content']}"
            for entry in context_entries
        ])

        # Try LLM-enhanced answer
        if self.ollama_available is None:
            await self.check_ollama()

        if self.ollama_available:
            try:
                return await self._llm_answer(question, context_text, context_entries)
            except Exception as e:
                logger.warning(f"LLM answer failed: {e}")

        # Fallback: return context directly
        return {
            "answer": context_text,
            "sources": [e["source"] for e in context_entries],
            "topics": [e["topic"] for e in context_entries],
            "method": "retrieval_only",
            "note": (
                "This information is from our knowledge base. For personalized "
                "medical advice, please consult a healthcare professional."
            ),
        }

    async def _llm_answer(
        self, question: str, context: str, entries: List[Dict]
    ) -> Dict:
        """Generate an LLM-enhanced answer using retrieved context."""
        import httpx

        prompt = f"""You are a knowledgeable health information assistant. Answer the user's question using ONLY the provided context. Do not add medical advice, diagnoses, or treatment recommendations beyond what is stated in the context.

Context:
{context}

Question: {question}

Rules:
- Answer based only on the provided context
- Use clear, non-technical language
- Do not diagnose any condition
- Do not recommend specific treatments
- Remind the user to consult a healthcare professional for personal medical concerns
- If the context doesn't cover the question, say so honestly"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "answer": result.get("response", ""),
                    "sources": [e["source"] for e in entries],
                    "topics": [e["topic"] for e in entries],
                    "method": "rag_llm",
                    "model": OLLAMA_MODEL,
                    "note": (
                        "This response was generated by an AI using our knowledge base. "
                        "It is for informational purposes only and should not be "
                        "considered medical advice."
                    ),
                }

        raise Exception("LLM request failed")


# Singleton
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
