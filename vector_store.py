# vector_store_db.py (simplified to work only with meds DB)
from typing import List, Dict, Any
from db import search_drug  # Function that queries the meds table in PostgreSQL

# ---------- General settings ----------
TOP_K = 5  # Number of results to return (placeholder, here we always return 1)

class RetrievedChunk:
    """
    Simple data container for a retrieved drug record.
    In a real vector store, this would hold similarity scores and embedding metadata.
    """
    def __init__(self, id: str, content: str, source: str):
        self.id = id
        self.content = content
        self.source = source

def query_similar(drug_name: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Replacement for a vector store similarity query.
    Instead of embeddings search, it fetches drug data directly from the meds table.
    Returns a list of dicts similar to what a vector DB would return.
    """
    data = search_drug(drug_name)
    if not data:
        return []

    # Format drug data into a readable text block
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

    # Return in vector-store-like format (id, content, metadata, score)
    return [{
        "id": data["drug_name"],
        "content": chunk_text,
        "metadata": {"source": "SafeMeds DB"},
        "score": None  # Score is not available since we are not using embeddings
    }]

# ---------- Usage example ----------
if __name__ == "__main__":
    results = query_similar("paracetamol")
    for r in results:
        print(r["content"])

