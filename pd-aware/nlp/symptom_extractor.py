"""
Symptom extraction from natural language input.

Uses keyword matching as a reliable baseline, with optional LLM enhancement
via Ollama for more nuanced understanding.

IMPORTANT: Extracted symptoms are ALWAYS shown to the user for review before
being used. The NLP output does not directly determine model predictions.
The voice-based model uses acoustic features, not symptom text.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Symptom Knowledge Base ──────────────────────────────────────────────────
# Each entry maps keywords → a structured symptom with metadata

SYMPTOM_PATTERNS = {
    "tremor": {
        "id": "tremor_rest",
        "name": "Tremor",
        "category": "motor",
        "description": "Involuntary shaking or trembling",
        "keywords": [
            "tremor", "shaking", "shaky", "shake", "trembling", "tremble",
            "quiver", "quivering", "vibrating", "jerky", "jerk",
        ],
        "body_parts": ["hand", "arm", "leg", "finger", "lip", "chin", "jaw"],
        "contexts": ["rest", "resting", "sitting", "relaxed", "still"],
    },
    "bradykinesia": {
        "id": "bradykinesia",
        "name": "Slowness of movement",
        "category": "motor",
        "description": "Movements feel slower than usual",
        "keywords": [
            "slow", "slower", "slowness", "sluggish", "bradykinesia",
            "takes longer", "difficulty starting", "hard to start",
            "frozen", "freezing", "stuck",
        ],
    },
    "rigidity": {
        "id": "rigidity",
        "name": "Muscle stiffness",
        "category": "motor",
        "description": "Muscles feel tight or resistant to movement",
        "keywords": [
            "stiff", "stiffness", "rigid", "rigidity", "tight", "tightness",
            "resistance", "hard to move", "locked", "muscle pain",
        ],
    },
    "gait_changes": {
        "id": "gait_changes",
        "name": "Changes in walking",
        "category": "motor",
        "description": "Difficulty walking or changes in gait",
        "keywords": [
            "walking", "walk", "gait", "shuffle", "shuffling", "dragging feet",
            "small steps", "short steps", "difficulty walking", "stumble",
            "limp", "unsteady", "festination",
        ],
    },
    "balance": {
        "id": "balance",
        "name": "Balance difficulties",
        "category": "motor",
        "description": "Feeling unsteady or having falls",
        "keywords": [
            "balance", "unsteady", "dizzy", "dizziness", "falling", "falls",
            "stumble", "postural", "tipping", "wobble", "sway",
        ],
    },
    "handwriting": {
        "id": "handwriting",
        "name": "Changes in handwriting",
        "category": "motor",
        "description": "Writing has become smaller or harder to control",
        "keywords": [
            "handwriting", "writing", "micrographia", "small writing",
            "smaller", "cramped", "letters", "pen", "pencil",
        ],
    },
    "facial_expression": {
        "id": "facial_expression",
        "name": "Reduced facial expression",
        "category": "motor",
        "description": "Face appears less animated",
        "keywords": [
            "facial expression", "face", "mask", "masked", "expressionless",
            "blank face", "stone face", "less expression", "flat face",
        ],
    },
    "speech_changes": {
        "id": "speech_changes",
        "name": "Speech changes",
        "category": "motor",
        "description": "Voice has become quieter or less clear",
        "keywords": [
            "speech", "voice", "soft voice", "quiet voice", "softer",
            "mumble", "mumbling", "slurred", "monotone", "hoarse",
            "hard to understand", "whisper",
        ],
    },
    "smell": {
        "id": "smell",
        "name": "Reduced sense of smell",
        "category": "non_motor",
        "description": "Difficulty identifying or noticing smells",
        "keywords": [
            "smell", "anosmia", "hyposmia", "can't smell", "lost smell",
            "reduced smell", "taste", "odor", "scent",
        ],
    },
    "constipation": {
        "id": "constipation",
        "name": "Constipation",
        "category": "non_motor",
        "description": "Ongoing bowel changes",
        "keywords": [
            "constipation", "constipated", "bowel", "digestive",
            "stomach", "bloating", "irregular bowel",
        ],
    },
    "sleep_changes": {
        "id": "sleep_changes",
        "name": "Sleep changes",
        "category": "non_motor",
        "description": "Sleep disturbances",
        "keywords": [
            "sleep", "insomnia", "restless", "dream", "acting out dreams",
            "rem", "rem sleep", "daytime sleepiness", "drowsy", "fatigue",
            "tired", "exhausted", "sleepy",
        ],
    },
    "mood_changes": {
        "id": "mood_changes",
        "name": "Mood changes",
        "category": "non_motor",
        "description": "Changes in mood or emotional state",
        "keywords": [
            "depression", "depressed", "anxiety", "anxious", "mood",
            "apathy", "motivation", "interest", "sad", "worried",
            "irritable", "emotional",
        ],
    },
    "fatigue": {
        "id": "fatigue",
        "name": "Fatigue",
        "category": "non_motor",
        "description": "Persistent tiredness",
        "keywords": [
            "fatigue", "tired", "exhausted", "energy", "weak",
            "weakness", "lethargy", "lethargic", "worn out",
        ],
    },
}


def extract_symptoms_keyword(text: str) -> List[Dict]:
    """
    Extract symptoms from text using keyword matching.

    This is the reliable baseline approach. Each matched symptom includes
    the original text snippet that triggered the match.
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    found_symptoms = []
    seen_ids = set()

    for symptom_key, symptom_info in SYMPTOM_PATTERNS.items():
        matched_keywords = []

        for keyword in symptom_info["keywords"]:
            # Check both word-level and phrase-level matches
            if " " in keyword:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
            else:
                if keyword in words:
                    matched_keywords.append(keyword)

        if matched_keywords and symptom_info["id"] not in seen_ids:
            seen_ids.add(symptom_info["id"])

            # Extract the context (surrounding text around the keyword)
            context = _extract_context(text, matched_keywords[0])

            found_symptoms.append({
                "id": symptom_info["id"],
                "name": symptom_info["name"],
                "category": symptom_info["category"],
                "description": symptom_info["description"],
                "matched_keywords": matched_keywords,
                "context": context,
                "confidence": "keyword_match",
                "source": "user_text",
            })

    return found_symptoms


def _extract_context(text: str, keyword: str, window: int = 50) -> str:
    """Extract text surrounding a keyword for context display."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return ""

    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)

    context = text[start:end]
    if start > 0:
        context = "..." + context
    if end < len(text):
        context = context + "..."

    return context


def extract_time_info(text: str) -> Dict:
    """Extract time-related information from text."""
    time_patterns = {
        "duration": [
            r"(\d+)\s*(day|week|month|year)s?\s*(ago)?",
            r"for\s*(the\s*last|about|around|over)?\s*(\d+)\s*(day|week|month|year)s?",
            r"since\s*(last\s*)?(january|february|march|april|may|june|july|august|september|october|november|december|\d{4})",
            r"recently",
            r"(a few|several|couple of)\s*(day|week|month|year)s?",
        ],
        "onset": [
            r"started?\s*(last|recently|suddenly|gradually)",
            r"began?\s*(last|recently|suddenly|gradually)",
            r"noticed?\s*(it)?\s*(last|recently|first|initially)",
        ],
        "progression": [
            r"(getting|becoming|growing)\s*(worse|better|more|less)",
            r"(worsening|improving|progressing|spreading)",
            r"(gradually|slowly|rapidly|quickly)\s*(getting|worsening|improving)",
        ],
    }

    results = {}
    text_lower = text.lower()

    for category, patterns in time_patterns.items():
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text_lower)
            if found:
                matches.extend([" ".join(m) if isinstance(m, tuple) else m for m in found])
        if matches:
            results[category] = matches

    return results


async def extract_symptoms_llm(
    text: str,
    ollama_base_url: str = "http://localhost:11434",
    model: str = "llama3.2",
) -> List[Dict]:
    """
    Use a local LLM via Ollama to extract symptoms from text.

    Falls back to keyword matching if Ollama is not available.

    IMPORTANT: The LLM does NOT make the final prediction. It only helps
    identify potential symptoms from the user's description. All extracted
    symptoms are shown to the user for review.
    """
    try:
        import httpx

        prompt = f"""You are a medical text analysis assistant. Extract symptoms from the following patient description.

For each symptom found, provide:
- symptom_name: A short name
- category: "motor" or "non_motor"  
- severity_hint: "mild", "moderate", "severe", or "unknown"
- relevant_quote: The exact text that mentions this symptom

Patient description:
"{text}"

Respond in JSON format only:
{{"symptoms": [{{"symptom_name": "...", "category": "...", "severity_hint": "...", "relevant_quote": "..."}}]}}

Important rules:
- Only extract symptoms the patient actually described
- Do not invent symptoms not mentioned in the text
- Do not diagnose any condition
- Use "unknown" severity when the patient doesn't indicate severity"""

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )

            if response.status_code == 200:
                import json
                result = response.json()
                llm_output = json.loads(result.get("response", "{}"))
                symptoms = llm_output.get("symptoms", [])

                # Convert to our format
                extracted = []
                for s in symptoms:
                    extracted.append({
                        "id": s.get("symptom_name", "").lower().replace(" ", "_"),
                        "name": s.get("symptom_name", "Unknown"),
                        "category": s.get("category", "unknown"),
                        "description": s.get("relevant_quote", ""),
                        "severity_hint": s.get("severity_hint", "unknown"),
                        "confidence": "llm_extracted",
                        "source": "ollama_llm",
                    })
                return extracted
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
                return extract_symptoms_keyword(text)

    except Exception as e:
        logger.warning(f"LLM extraction failed ({e}), falling back to keyword matching")
        return extract_symptoms_keyword(text)


def merge_extractions(
    keyword_results: List[Dict],
    llm_results: List[Dict],
) -> List[Dict]:
    """Merge keyword and LLM extraction results, preferring LLM enrichment."""
    merged = {}

    # Start with keyword results (reliable baseline)
    for item in keyword_results:
        merged[item["id"]] = item.copy()
        merged[item["id"]]["extraction_methods"] = ["keyword"]

    # Enhance with LLM results
    for item in llm_results:
        item_id = item["id"]
        if item_id in merged:
            # Enhance existing entry
            merged[item_id]["extraction_methods"].append("llm")
            if "severity_hint" in item:
                merged[item_id]["severity_hint"] = item["severity_hint"]
        else:
            # New symptom from LLM
            item["extraction_methods"] = ["llm"]
            merged[item_id] = item

    return list(merged.values())
