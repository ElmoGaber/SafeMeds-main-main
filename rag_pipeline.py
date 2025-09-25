# rag_core.py (modified to work with DB only)
from typing import List, Dict, Any
from dataclasses import dataclass
from db import search_drug  # Function that fetches drug data from PostgreSQL

# ---------- General settings ----------
TOP_K = 5  # Number of reference results (in case we want multiple)

# System prompt metadata (for reference if connected to LLMs later)
_SYSTEM_PROMPT = {
    "name": "SafeMeds Assistant",
    "role": "Medical Information Assistant",
    "instructions": (
        "You are SafeMeds Assistant, a trusted medical information helper. "
        "Your main goal is to answer questions related to drugs after sleeve gastrectomy, "
        "based only on verified information from the database. "
        "If the available information is not enough, clearly tell the user. "
        "Use clear, accurate, and simple language."
    )
}

@dataclass
class RetrievedChunk:
    """
    Data structure for holding a retrieved piece of information.
    """
    id: str
    text: str
    source: str
    score: float | None = None  # Can be used if vector similarity is added later

# ---------- Data retrieval from DB ----------
def retrieve_context(drug_name: str, top_k: int = TOP_K) -> List[RetrievedChunk]:
    """
    Retrieve drug information from the database and return it as RetrievedChunk objects.
    Currently fetches by exact drug name (via search_drug).
    """
    data = search_drug(drug_name)
    if not data:
        return []

    # Format all drug fields into a structured text block
    chunk_text = (
        f"Drug Name: {data['drug_name']}\n"
        f"Generic: {data['generic_name']}\n"
        f"Class: {data['drug_class']}\n"
        f"Indication: {data['indication']}\n"
        f"Status After Sleeve: {data['status_after_sleeve']}\n"
        f"Reason: {data['reason']}\n"
        f"Dose Adjustment: {data['dose_adjustment_notes']}\n"
        f"Form: {data['administration_form']}\n"
        f"Notes: {data['notes']}\n"
    )

    return [RetrievedChunk(
        id=data['drug_name'],
        text=chunk_text,
        source="SafeMeds DB"
    )]

# ---------- Answer generation ----------
def answer_question(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Main function to answer a user question.
    - Retrieves relevant context from DB
    - If nothing is found, returns a fallback message
    - Otherwise, compiles the drug information into an answer
    """
    chunks = retrieve_context(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "لا توجد معلومات كافية في قاعدة البيانات للإجابة على سؤالك.",  # fallback in Arabic
            "citations": [],
            "used_chunks": []
        }

    # Simplest answer: concatenate the retrieved chunks
    raw_answer = "\n".join(c.text for c in chunks)

    # Simple citation format: [1], [2], ... based on order
    citations = [f"[{i}]" for i in range(1, len(chunks) + 1)]

    return {
        "answer": raw_answer,
        "citations": citations,
        "used_chunks": [c.__dict__ for c in chunks]
    }
