import sqlite3  # fallback لو عايز SQLite لاحقًا
from datetime import date
import csv
import psycopg
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import re
import logging

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------
# إعدادات قاعدة البيانات
# -------------------------------
DB_PATH = "bariatric_meds.db"  # fallback لـ SQLite لو عايز

POSTGRES_URL = "postgresql://postgres:12345@localhost:5432/postgres"  # غيّر الباسورد لو مختلف

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS meds (
    id SERIAL PRIMARY KEY,
    drug_name TEXT UNIQUE,
    generic_name TEXT,
    drug_class TEXT,
    indication TEXT,
    status_after_sleeve TEXT,
    reason TEXT,
    dose_adjustment_notes TEXT,
    administration_form TEXT,
    interactions TEXT,
    evidence_level TEXT,
    source_links TEXT,
    last_reviewed DATE,
    notes TEXT,
    embedding JSONB
);
"""

EXAMPLE_ENTRIES = [
    {
        "drug_name": "Ibuprofen",
        "generic_name": "ibuprofen",
        "drug_class": "NSAID",
        "indication": "مسكن/مضاد التهاب",
        "status_after_sleeve": "avoid",
        "reason": "يزيد خطر التقرحات بعد جراحات السمنة",
        "dose_adjustment_notes": "تجنب نهائيًا إن أمكن؛ استخدم باراسيتامول كبديل",
        "administration_form": "tablet",
        "interactions": "",
        "evidence_level": "guideline/review",
        "source_links": "https://www.sps.nhs.uk/articles/considerations-for-using-medicines-following-bariatric-surgery/",
        "last_reviewed": str(date.today()),
        "notes": "ممنوع في معظم الحالات بعد التكميم."
    },
    {
        "drug_name": "Omeprazole",
        "generic_name": "omeprazole",
        "drug_class": "PPI",
        "indication": "حماية من القرحة/علاج الارتجاع",
        "status_after_sleeve": "conditional",
        "reason": "يُستخدم كوقاية بعد الجراحة لفترة معينة",
        "dose_adjustment_notes": "عادة يُوصف لشهور بعد العملية",
        "administration_form": "capsule",
        "interactions": "",
        "evidence_level": "guideline",
        "source_links": "https://pubmed.ncbi.nlm.nih.gov/",
        "last_reviewed": str(date.today()),
        "notes": "المدة تختلف حسب تعليمات الجرّاح."
    },
    {
        "drug_name": "Paracetamol",
        "generic_name": "acetaminophen",
        "drug_class": "Analgesic",
        "indication": "مسكن ألم",
        "status_after_sleeve": "allowed",
        "reason": "آمن كمسكن بعد التكميم",
        "dose_adjustment_notes": "انتبه للجرعة القصوى اليومية",
        "administration_form": "tablet/liquid",
        "interactions": "",
        "evidence_level": "guideline",
        "source_links": "https://www.ssmhealth.com/",
        "last_reviewed": str(date.today()),
        "notes": "البديل المفضل للمسكنات بعد التكميم."
    }
]

# -------------------------------
# PostgreSQL Connection
# -------------------------------
def get_connection():
    try:
        conn = psycopg.connect(POSTGRES_URL)
        logging.info("✅ Connected to PostgreSQL.")
        return conn
    except Exception as e:
        logging.error(f"DB Connection error: {e}. تأكد إن PostgreSQL شغال و الـ URL صح.")
        return None

def create_postgres_table():
    conn = get_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_POSTGRES)
        conn.commit()
        logging.info("✅ Table 'meds' created or exists.")
        return True
    except Exception as e:
        logging.error(f"Table creation error: {e}")
        return False
    finally:
        conn.close()

# -------------------------------
# Embeddings (معدل: model multilingual للعربي)
# -------------------------------
# حمل الـ model (أول مرة هياخد وقت)
try:
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    logging.info("✅ Multilingual embedding model loaded.")
except Exception as e:
    logging.error(f"Model load error: {e}. جرب pip install sentence-transformers")
    model = None

def get_embeddings(texts):
    if not model:
        return [[] for _ in texts]  # fallback لو مش محمل
    # Normalize: lowercase وإزالة إشارات لتحسين الـ similarity
    normalized = [re.sub(r'[^\w\s]', '', t.lower().strip()) for t in texts]
    return model.encode(normalized, convert_to_numpy=True).tolist()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# -------------------------------
# إدخال / بحث (معدل)
# -------------------------------
def insert_drug(entry: dict):
    defaults = {
        "drug_class": "",
        "status_after_sleeve": "",
        "reason": "",
        "dose_adjustment_notes": "",
        "administration_form": "",
        "interactions": "",
        "evidence_level": "",
        "source_links": "",
        "last_reviewed": date.today(),
        "notes": ""
    }
    for k, v in defaults.items():
        entry.setdefault(k, v)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cols = ",".join(entry.keys())
            placeholders = ",".join(["%s"] * len(entry))
            sql = f"""
                INSERT INTO meds ({cols}) VALUES ({placeholders})
                ON CONFLICT (drug_name) DO UPDATE SET
                    generic_name = EXCLUDED.generic_name,
                    indication = EXCLUDED.indication,
                    notes = EXCLUDED.notes,
                    last_reviewed = EXCLUDED.last_reviewed
                RETURNING id;
            """
            cur.execute(sql, tuple(entry.values()))
            result = cur.fetchone()

            if result:
                drug_id = result[0]
                text = f"{entry['drug_name']} ({entry.get('generic_name','')}) - {entry.get('indication','')}"
                emb = get_embeddings([text])[0]
                cur.execute(
                    "UPDATE meds SET embedding = %s WHERE id = %s;",
                    (json.dumps(emb), drug_id)
                )
                logging.info(f"✅ Drug '{entry['drug_name']}' inserted/updated with embedding.")
            else:
                logging.warning(f"⚠️ Drug '{entry['drug_name']}' already exists. Updated fields.")

        # مهم: نعمل commit بعد ما نقفل الكيرسور
        conn.commit()
        logging.info(f"💾 Drug '{entry['drug_name']}' saved to DB successfully.")
        return True

    except Exception as e:
        logging.error(f"Insert error: {e}")
        return False

    finally:
        conn.close()

def search_drug(query: str, top_k: int = 3):
    if not query.strip():
        return []

    conn = get_connection()
    if not conn:
        logging.error("Cannot search: No DB connection.")
        return []

    # 1️⃣ Exact/Partial match fallback (SQL LIKE – أسرع وأدق للأسماء)
    try:
        with conn.cursor() as cur:
            # بحث جزئي على drug_name أو generic_name (case-insensitive)
            cur.execute("""
                SELECT drug_name, generic_name, drug_class, indication,
                       status_after_sleeve, reason, dose_adjustment_notes,
                       administration_form, interactions, evidence_level,
                       source_links, notes, embedding
                FROM meds 
                WHERE LOWER(drug_name) LIKE LOWER(%s) OR LOWER(generic_name) LIKE LOWER(%s)
                LIMIT %s;
            """, (f"%{query}%", f"%{query}%", top_k))
            exact_rows = cur.fetchall()
        
        if exact_rows:
            results = []
            for row in exact_rows:
                emb = json.loads(row[12]) if row[12] else []
                results.append({
                    "drug_name": row[0], "generic_name": row[1], "drug_class": row[2],
                    "indication": row[3], "status_after_sleeve": row[4], "reason": row[5],
                    "dose_adjustment_notes": row[6], "administration_form": row[7],
                    "interactions": row[8], "evidence_level": row[9], "source_links": row[10],
                    "notes": row[11],
                })
            logging.info(f"✅ Exact search for '{query}': {len(results)} results.")
            conn.close()
            return results
    except Exception as e:
        logging.error(f"Exact search error: {e}")

    # 2️⃣ Semantic search (لو مفيش exact)
    try:
        query_emb = get_embeddings([query])[0]
        with conn.cursor() as cur:
            cur.execute("""
                SELECT drug_name, generic_name, drug_class, indication,
                       status_after_sleeve, reason, dose_adjustment_notes,
                       administration_form, interactions, evidence_level,
                       source_links, notes, embedding
                FROM meds WHERE embedding IS NOT NULL;
            """)
            rows = cur.fetchall()

        scored = []
        for row in rows:
            emb_str = row[12]
            if isinstance(emb_str, str):
                emb = json.loads(emb_str)
            else:
                emb = emb_str or []

            if len(emb) > 0:  # تجاهل لو embedding فاضي
                score = cosine_similarity(query_emb, emb)
                if score > 0.3:  # threshold لنتائج ذات صلة (عدّل لو عايز أوسع)
                    scored.append((score, {
                        "drug_name": row[0], "generic_name": row[1], "drug_class": row[2],
                        "indication": row[3], "status_after_sleeve": row[4], "reason": row[5],
                        "dose_adjustment_notes": row[6], "administration_form": row[7],
                        "interactions": row[8], "evidence_level": row[9], "source_links": row[10],
                        "notes": row[11],
                    }))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:top_k]]
        logging.info(f"🔍 Semantic search for '{query}': {len(results)} results (top scores: {[(round(s[0], 2)) for s in scored[:3]]})")
        conn.close()
        return results
    except Exception as e:
        logging.error(f"Semantic search error: {e}")
        conn.close()
        return []

def get_all_drugs():
    conn = get_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT drug_name, generic_name, indication, notes FROM meds;")
            rows = cur.fetchall()
        # حوّل لـ dicts للتوافق
        results = [{"drug_name": r[0], "generic_name": r[1], "indication": r[2], "notes": r[3]} for r in rows]
        logging.info(f"📂 Loaded {len(results)} drugs from DB.")
        conn.close()
        return results
    except Exception as e:
        logging.error(f"Get all drugs error: {e}")
        conn.close()
        return []

# -------------------------------
# إدخال يدوي من المستخدم (نفس الأصلي، بس مع logging)
# -------------------------------
def add_drug_direct(user_input: str):
    entry = {
        "drug_name": None,
        "generic_name": None,
        "drug_class": None,
        "indication": None,
        "status_after_sleeve": None,
        "reason": None,
        "dose_adjustment_notes": None,
        "administration_form": None,
        "interactions": None,
        "evidence_level": None,
        "source_links": None,
        "last_reviewed": str(date.today()),
        "notes": None
    }

    m = re.search(r"(?:اسمه|اسم الدواء)\s*([\w\d\+\-\s\u0600-\u06FF]+)", user_input, re.UNICODE)
    if m: entry["drug_name"] = m.group(1).strip()

    m = re.search(r"(?:المادة الفعالة|الاسم العلمي)\s*([\w\d\+\-\s\u0600-\u06FF]+)", user_input, re.UNICODE)
    if m: entry["generic_name"] = m.group(1).strip()

    m = re.search(r"(?:لعلاج|لـ|في)\s*([\w\d\+\-\s\u0600-\u06FF]+)", user_input, re.UNICODE)
    if m: entry["indication"] = m.group(1).strip()

    m = re.search(r"(?:ملاحظات|ملاحظة)\s*([\w\d\+\-\s\u0600-\u06FF]+)", user_input, re.UNICODE)
    if m: entry["notes"] = m.group(1).strip()

    clean_entry = {k: v for k, v in entry.items() if v is not None}

    if not clean_entry.get("drug_name"):
        logging.warning("⚠️ مش قادر أستخرج اسم الدواء من النص.")
        return False

    success = insert_drug(clean_entry)
    if success:
        logging.info(f"✅ {clean_entry['drug_name']} اتسجل في قاعدة البيانات.")
    return success

# -------------------------------
# تشغيل رئيسي (معدل: بدون duplicate)
# -------------------------------
if __name__ == "__main__":
    # إنشاء الجدول
    create_postgres_table()

    # إدخال الأمثلة مرة واحدة فقط لو الجدول فاضي
    all_drugs = get_all_drugs()
    if not all_drugs:
        logging.info("📝 Inserting example entries...")
        for e in EXAMPLE_ENTRIES:
                  insert_drug(e["drug_name"], e["generic_name"], e["indication"], e["notes"])