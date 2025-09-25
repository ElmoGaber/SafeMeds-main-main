import os
import requests
import time
import logging
import hashlib
import re
from typing import Dict, List
from prompting import prompt_config   # ملف إعدادات البرومبت
from db import search_drug, insert_drug, create_postgres_table  # دوال قاعدة البيانات (أضفت create_postgres_table)

# ======================
# Logging Setup
# ======================
logging.basicConfig(level=logging.INFO)

# ======================
# Gemini API Settings
# ======================
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB5m-avB33bwSfEwogIoBMJYdgwR8Yaw3s")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ======================
# Improved Cache Layer (مع expiration)
# ======================
class SimpleCache:
    def __init__(self):
        self.cache: Dict[str, tuple] = {}  # (response, timestamp)

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()

    def get(self, query: str, max_age=300):  # 5 دقائق صلاحية
        key = self._hash(query)
        if key in self.cache:
            resp, ts = self.cache[key]
            if time.time() - ts < max_age:
                return resp
            else:
                del self.cache[key]  # انتهت الصلاحية
                logging.info(f"Cache expired for query: {query}")
        return None

    def set(self, query: str, response: str):
        key = self._hash(query)
        self.cache[key] = (response, time.time())
        logging.info(f"Cache set for query: {query}")

cache = SimpleCache()

# ======================
# Symptom Keywords (نفس الأصلي)
# ======================
SYMPTOMS_KEYWORDS = {
    "وجع المعدة": ["وجع", "بطن", "معدة", "حرقة", "ألم في البطن"],
    "صداع": ["صداع", "راس وجع", "وجع راس"],
    "قيء": ["ترجيع", "سخنية", "غثيان"],
    "إسهال": ["اسهال", "إسهال", "إسهال مائي"],
    "تعب": ["تعب", "ارهاق", "ضعف"],
    "تنميل": ["تنميل", "وخز", "خدر"],
}

# ======================
# Mapping Symptoms ↔ Drugs (نفس الأصلي)
# ======================
SYMPTOM_TO_DRUG = {
    "وجع المعدة": ["Omeprazole", "Pantoprazole", "Ranitidine"],
    "صداع": ["Paracetamol", "Ibuprofen", "Aspirin"],
    "قيء": ["Metoclopramide", "Domperidone"],
    "إسهال": ["Loperamide", "Oral Rehydration Salt"],
    "تعب": ["Multivitamins", "Iron Supplement", "Vitamin B Complex"],
    "تنميل": ["Vitamin B Complex", "Magnesium"]
}

# ======================
# User State Management (نفس الأصلي، بس أضفت logging)
# ======================
USER_STATES = {}
ADD_FIELDS = ["drug_name", "generic_name", "indication", "notes"]
FIELD_PROMPTS = {
    "drug_name": "تمام 👍، قوللي اسم الدوا التجاري",
    "generic_name": "طب إيه المادة الفعالة أو الاسم العلمي؟ لو مش عارف قول *مش عارف*.",
    "indication": "طيب بيتاخد في إيه أو استخدامه إيه؟",
    "notes": "تحب تضيف أي ملاحظات زيادة (زي الجرعة، التحذيرات، السن المناسب)؟"
}

def add_drug_step_by_step(user_id: str, message: str):
    if user_id not in USER_STATES:
        USER_STATES[user_id] = {"data": {}, "next_field": 0}

    state = USER_STATES[user_id]
    idx = state["next_field"]

    if idx > 0:
        prev_field = ADD_FIELDS[idx - 1]
        state["data"][prev_field] = message.strip()

    if idx >= len(ADD_FIELDS):
        try:
            insert_drug(state["data"])
            logging.info(f"Inserted drug: {state['data']['drug_name']}")
            # أهم تعديل: امسح الـ Cache بعد الإضافة عشان يتحدث
            cache.cache.clear()
            logging.info("Cache cleared after drug insertion.")
            USER_STATES.pop(user_id)
            return f"✅ تمت إضافة الدواء '{state['data']['drug_name']}' بنجاح! الآن جرب تسأل عنه تاني."
        except Exception as e:
            logging.error(f"Insert error in add_drug: {e}")
            return f"⚠️ خطأ في حفظ الدواء: {e}. جرب تاني."

    next_field = ADD_FIELDS[idx]
    state["next_field"] += 1
    return FIELD_PROMPTS[next_field]

# ======================
# Match Symptom (نفس الأصلي)
# ======================
def match_symptom(user_input: str):
    user_input = user_input.lower()
    for symptom, keywords in SYMPTOMS_KEYWORDS.items():
        for kw in keywords:
            if kw in user_input:
                return symptom
    return None

# ======================
# Main Chat Wrapper (معدل بشكل كبير)
# ======================
def gemini_chat_wrapper(message: str, history: List = [], user_id: str = "default_user"):
    # 0️⃣ Add drug step-by-step (نفس الأصلي)
    if user_id in USER_STATES:
        return add_drug_step_by_step(user_id, message)

    if any(word in message for word in ["ضيف", "اضيف", "أدخل", "اضافة"]):
        return add_drug_step_by_step(user_id, "")

    # 1️⃣ Check if user asks about taking a drug (حسّنت الـ regex للعربي)
    take_drug_patterns = [
        r"هل ينفع آخد ([\w\d\s\u0600-\u06FF\u0750-\u077F]+)\??",  # دعم عربي (Unicode)
        r"ممكن آخد ([\w\d\s\u0600-\u06FF\u0750-\u077F]+)\??",
        r"ينفع آخد ([\w\d\s\u0600-\u06FF\u0750-\u077F]+)\??",
        r"هل يمكن أخذ ([\w\d\s\u0600-\u06FF\u0750-\u077F]+)\??",
    ]
    for pattern in take_drug_patterns:
        match = re.search(pattern, message, re.IGNORECASE | re.UNICODE)
        if match:
            drug_name_query = match.group(1).strip()
            logging.info(f"Extracted drug query: {drug_name_query}")
            drugs_in_db = search_drug(drug_name_query)
            if drugs_in_db:
                first = drugs_in_db[0]
                answer = f"✅ أه ينفع تاخد {first.get('drug_name', drug_name_query)}"
                missing = [f for f in ["generic_name", "indication", "drug_class"] if not first.get(f) or first.get(f) == "غير محدد"]
                if missing:
                    answer += f"\nℹ️ بعض التفاصيل ناقصة: {', '.join(missing)}. ممكن تضيفها عشان أقدر أفيدك أكتر."
                # حفظ في Cache
                cache.set(message, answer)
                return answer
            else:
                answer = f"⚠️ الدواء '{drug_name_query}' مش موجود في قاعدة البيانات. تحب تضيفه؟"
                cache.set(message, answer)
                return answer

    # 2️⃣ DB search first (تعديل رئيسي: قبل Cache!)
    drug_info = search_drug(message)
    logging.info(f"DB search for '{message}': {len(drug_info)} results")
    if drug_info:
        # بني relevant_context من كل النتائج (top 3)
        context_parts = []
        for info in drug_info:
            context_parts.append(f"""
اسم الدواء: {info.get('drug_name', 'غير محدد')}
الاسم العلمي: {info.get('generic_name', 'غير محدد')}
الفئة الدوائية: {info.get('drug_class', 'غير محدد')}
دواعي الاستعمال: {info.get('indication', 'غير محدد')}
الحالة بعد التكميم: {info.get('status_after_sleeve', 'غير محدد')}
السبب: {info.get('reason', 'غير محدد')}
ملاحظات تعديل الجرعة: {info.get('dose_adjustment_notes', 'لا توجد')}
شكل الدواء: {info.get('administration_form', 'غير محدد')}
ملاحظات عامة: {info.get('notes', 'لا توجد')}
""")
        relevant_context = "\n".join(context_parts)
        # استخدم Gemini مع الـ Context، أو رد مباشر لو بسيط
        # هنا، بني prompt وارسل لـ Gemini لإجابة مفصلة
    else:
        # لو مش دواء، جرب symptom
        symptom = match_symptom(message)
        if symptom:
            meds_list = SYMPTOM_TO_DRUG.get(symptom, [])
            meds_info = []
            for med in meds_list:
                info = search_drug(med)
                if info:
                    for i in info:
                        meds_info.append(f"{i.get('drug_name')} ({i.get('generic_name')}): {i.get('indication')}")
            if meds_info:
                relevant_context = f"بالنسبة لـ {symptom}، الأدوية المناسبة:\n- " + "\n- ".join(meds_info)
            else:
                relevant_context = f"واضح إن عندك {symptom}. حاول ترتاح وتشرب سوائل، واستشير دكتور لو الأعراض مستمرة."
        else:
            relevant_context = "❌ مفيش تطابق مباشر في قاعدة البيانات."

    # 3️⃣ Cache lookup (بعد DB، عشان لو DB لقى، مش هيروح هنا)
    if not drug_info and relevant_context == "❌ مفيش تطابق مباشر في قاعدة البيانات.":
        cached = cache.get(message)
        if cached:
            return cached

    # 4️⃣ Gemini call (لو وصل هنا، بني الـ prompt)
    history_text = "\n".join([
        f"{h['role']}: {h['message']}" if isinstance(h, dict) else f"{h[0]}: {h[1]}"
        for h in history
    ])

    final_prompt = f"""
{prompt_config['instructions']}

HISTORY:
{history_text}

CONTEXT:
{relevant_context}

QUESTION:
{message}
"""

    payload = {"contents": [{"parts": [{"text": final_prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error: {e}")
        return f"❌ Network error: {e}"

    if response.status_code == 200:
        data = response.json()
        try:
            answer = data["candidates"][0]["content"]["parts"][0]["text"]
            # حفظ في Cache لو مش من DB (عشان الـ DB محفوظ بالفعل)
            if not drug_info:
                cache.set(message, answer)
            return answer
        except Exception as e:
            logging.error(f"Unexpected response format: {e}")
            return f"⚠️ Unexpected response format: {e}\n{data}"
    else:
        logging.error(f"API error: {response.status_code} - {response.text}")
        return f"❌ Error: {response.status_code} - {response.text}"

# ======================
# Logging (نفس الأصلي)
# ======================
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # أضف init للـ DB
    create_postgres_table()
    
    start = time.time()
    user_id = "local_user"
    message = input("اكتب رسالتك: ")
    print(gemini_chat_wrapper(message, user_id=user_id))
