# filename: ingest_meds.py
import psycopg
from sentence_transformers import SentenceTransformer

# Database configuration
DB_URL = "postgresql://postgres:12345@localhost:5432/postgres"

# Load the embedding model from HuggingFace
model = SentenceTransformer("BAAI/bge-small-en")

# -------------------------------
# DB utils
# -------------------------------
def get_connection():
    """
    Create and return a new connection to the PostgreSQL database
    """
    return psycopg.connect(DB_URL)

def add_embeddings_column():
    """
    Add an embeddings column to the 'meds' table if it does not already exist.
    The column type is VECTOR(384) (requires pgvector extension).
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name='meds' AND column_name='embedding'
            ) THEN
                ALTER TABLE meds ADD COLUMN embedding VECTOR(384);
            END IF;
        END$$;
    """)
    conn.commit()
    conn.close()

# -------------------------------
# Ingest from DB
# -------------------------------
def ingest_meds():
    """
    Fetch all medicines data from the 'meds' table.
    Combine drug_name, indication, and notes into a single text string.
    Returns a list of tuples (id, text).
    """
    all_texts = []
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, drug_name, indication, notes FROM meds;")
    rows = cur.fetchall()

    for row in rows:
        med_id, drug_name, indication, notes = row
        # Join available fields into one text representation
        text = " ".join([drug_name or "", indication or "", notes or ""]).strip()
        if not text:
            continue
        all_texts.append((med_id, text))

    print(f"📋 Loaded {len(all_texts)} meds from DB")
    conn.close()
    return all_texts

def update_embeddings():
    """
    Generate embeddings for each medicine using the SentenceTransformer model.
    Update the 'embedding' column in the DB with the generated vectors.
    """
    conn = get_connection()
    cur = conn.cursor()

    meds = ingest_meds()
    for med_id, text in meds:
        emb = model.encode([text])[0]  # numpy array
        cur.execute("UPDATE meds SET embedding = %s WHERE id = %s;", (emb.tolist(), med_id))

    conn.commit()
    conn.close()
    print(f"✅ Updated embeddings for {len(meds)} meds")

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    # Step 1: Ensure the embedding column exists
    add_embeddings_column()
    # Step 2: Generate and update embeddings for all medicines
    update_embeddings()


    