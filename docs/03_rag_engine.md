# 03 RAG Engine (เครื่องยนต์ระบบดึงความรู้)

เอกสารฉบับนี้เจาะลึกระบบ **Retrieval-Augmented Generation (RAG)** ซึ่งเป็นหัวใจสำคัญที่ทำให้ AI "ตอบตามข้อเท็จจริง" ได้ แทนที่จะ "แต่งเรื่องขึ้นมาเอง"

---

## 1. RAG คืออะไร? (ปูพื้นฐาน)

### ปัญหาของ LLM ที่ไม่มี RAG
โมเดลภาษาถูกเทรนมาจากข้อมูลในอินเทอร์เน็ต มันรู้เรื่องทั่วไป แต่ **ไม่รู้เนื้อหาในหนังสือส่วนตัวของคุณเลย**

```
❌ ไม่มี RAG:
ผู้ใช้: "หนังสือ 'คิดแบบวิศวกร' สอนอะไร?"
AI: "ผมไม่ทราบครับ เพราะไม่เคยอ่านหนังสือเล่มนี้"
   (หรือร้ายกว่านั้น: แต่งเรื่องขึ้นมาเอง!)

✅ มี RAG:
ผู้ใช้: "หนังสือ 'คิดแบบวิศวกร' สอนอะไร?"
AI: [ค้นหาในฐานข้อมูล → พบเนื้อหาที่เกี่ยวข้อง → ส่งให้ LLM]
AI: "หนังสือ 'คิดแบบวิศวกร' เน้นสอนเรื่อง..."
```

### RAG Pipeline ภาพรวม
```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  1. PREPARE     │     │  2. RETRIEVE    │     │  3. GENERATE   │
│  (Offline)      │     │  (Runtime)      │     │  (Runtime)     │
│                 │     │                 │     │                │
│  หนังสือ        │     │  คำถามผู้ใช้      │     │  LLM + Context │
│     ↓           │     │     ↓           │     │     ↓          │
│  แตกเป็นชิ้น    │     │  แปลง → Vector  │     │  สร้างคำตอบ     │
│     ↓           │     │     ↓           │     │  ที่อ้างอิง     │
│  แปลง → Vector │     │  ค้นหาใน FAISS  │     │  จากหนังสือ     │
│     ↓           │     │     ↓           │     │                │
│  เก็บใน FAISS  │     │  ได้ Context     │────▶│                │
└────────────────┘     └────────────────┘     └────────────────┘
  build_index.py         rag_engine.py           ai_core.py
```

---

## 2. Vector Embeddings — คณิตศาสตร์ของความหมาย

### แนวคิดพื้นฐาน
Embedding คือการแปลง "คำ" ให้เป็น "ตัวเลข" ที่คอมพิวเตอร์เข้าใจ โดยรักษาความหมายไว้

```
ตัวอย่าง (สมมติลดรูปเหลือ 3 มิติ):
"หมา"   → [0.9, 0.1, 0.3]
"สุนัข"  → [0.88, 0.12, 0.28]  ← ใกล้กับ "หมา" มาก! (คำเดียวกัน)
"แมว"   → [0.7, 0.3, 0.35]   ← ใกล้พอควร (สัตว์เลี้ยงเหมือนกัน)
"รถยนต์" → [0.1, 0.8, 0.9]   ← ไกลมาก (ไม่เกี่ยวกัน)
```

ในความเป็นจริง โมเดล `BAAI-bge-m3` สร้าง Vector ขนาด **1024 มิติ** (ไม่ใช่แค่ 3) ทำให้สามารถจับความแตกต่างของความหมายได้ละเอียดมากๆ

### โค้ดที่ใช้สร้าง Embedding:
```python
# build_index.py → build_and_save_index()
embedding_text = f"query: จากหนังสือ '{book}', เนื้อหา: {content}"
embeddings = self.model.encode(
    texts_to_embed,
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")
```

**Best Practice: Prefix "query:" สำคัญมาก!**
```python
# ✅ GOOD: ใส่ prefix "query:" ก่อนข้อความ
embedding_text = f"query: จากหนังสือ '{book}', เนื้อหา: {content}"

# ❌ BAD: ส่งข้อความเปล่าๆ
embedding_text = content
```
**ทำไม?** โมเดล `bge-m3` ถูกเทรนมาให้คาดหวัง prefix `query:` ถ้าไม่ใส่ ค่า embedding จะ "ออกทะเล" ไม่ตรงกับความหมายจริง

---

## 3. FAISS — ฐานข้อมูลในหน่วยความจำ

### FAISS คืออะไร?
**Facebook AI Similarity Search (FAISS)** คือไลบรารีที่ Meta สร้างขึ้น เพื่อค้นหา Vector ที่ "ใกล้เคียง" กันมากที่สุดด้วยความเร็วสูงมาก

### การสร้าง Index (build_index.py)
```python
# สร้าง Index แบบ FlatL2 (Brute-force, แม่นยำ 100%)
index = faiss.IndexFlatL2(embeddings.shape[1])  # shape[1] = 1024 มิติ
index.add(embeddings)  # ยัดเวกเตอร์ทั้งหมดเข้าไป
faiss.write_index(index, "faiss_index.bin")  # เซฟเป็นไฟล์
```

### ประเภทของ FAISS Index:
| Type | ความแม่นยำ | ความเร็ว | เหมาะกับ |
|------|-----------|----------|----------|
| `IndexFlatL2` ✅ (ที่เราใช้) | 100% (Exact) | ช้าสุด | ข้อมูลน้อย (หลักพัน records) |
| `IndexIVFFlat` | ~95-98% | เร็วกว่า 10x | ข้อมูลปานกลาง (หลักแสน) |
| `IndexHNSW` | ~95-99% | เร็วที่สุด | ข้อมูลมาก (หลักล้าน) |

**ทำไมเราใช้ FlatL2?**
เพราะฐานข้อมูลหนังสือของเรามีเพียงหลักร้อยถึงหลักพัน records
การค้นหาแบบ Brute-force ใช้เวลาเพียง ~1-5ms ซึ่งเร็วเพียงพอ และได้ความแม่นยำ 100%!

---

## 4. Search & Retrieval Pipeline (ขั้นตอนการค้นหาแบบละเอียด)

เมื่อผู้ใช้ถาม "ความอดทนสำคัญอย่างไร?" ระบบทำงานดังนี้:

### Step 1: Encode Query (แปลงคำถามเป็น Vector)
```python
# rag_engine.py → search()
query_vector = self.embedder.encode(
    ["query: " + query],    # ใส่ prefix "query:"
    convert_to_numpy=True
)
# ผลลัพธ์: array ขนาด [1, 1024] — 1 คำถาม, 1024 มิติ
```
**สำคัญ:** Embedder ทำงานบน **CPU** (ไม่ใช่ GPU) 
เพราะ GPU ถูกจองไว้ให้โมเดลภาษาทั้งหมด (ดูรายละเอียดในเอกสาร 04)

### Step 2: FAISS Search (ค้นหา Top-K ที่ใกล้เคียง)
```python
retrieval_count = top_k * 3  # ค้นหา 18 candidates (6 × 3)
distances, indices = self.index.search(query_vector, retrieval_count)
```
**ทำไมค้นหา 18 ตัว แต่เอาแค่ 6?**
เพราะในขั้นตอนถัดไปจะมีการ Deduplicate (ลบซ้ำ) ดังนั้นต้องดึงมาเผื่อให้เยอะพอ

### Step 3: Distance → Similarity Score (แปลงระยะทางเป็นคะแนน)
```python
# L2 Distance: ยิ่งน้อย = ยิ่งใกล้ = ยิ่งเกี่ยวข้อง
# แต่คนทั่วไปเข้าใจ "ยิ่งมาก = ยิ่งดี" ง่ายกว่า
# จึงแปลงด้วยสูตร:
similarity = 1 / (1 + distance)
```
**ตัวอย่างตัวเลข:**
| Distance (L2) | Similarity Score | ความหมาย |
|---------------|-----------------|----------|
| 0.0 | 1.00 | เหมือนกันเป๊ะ (เป็นไปไม่ได้ในทางปฏิบัติ) |
| 0.5 | 0.67 | เกี่ยวข้องมาก ✅ |
| 1.0 | 0.50 | ปานกลาง ⚠️ |
| 5.0 | 0.17 | ไม่เกี่ยวข้องเลย ❌ |

### Step 4: Deduplication (กำจัดข้อมูลซ้ำ)
```python
def _deduplicate_passages(self, results, min_diff_len=50):
    seen = set()
    unique_results = []
    for score, item in results:
        content = item.get("content", "").strip()
        key = (item.get("book_title"), content[:min_diff_len])  # ① ใช้ 50 ตัวอักษรแรก
        if key not in seen:  # ② O(1) lookup ใน Set
            seen.add(key)
            unique_results.append((score, item))
    return unique_results
```

**ทำไมต้อง Deduplicate?**
สมมติหนังสือเล่มหนึ่งมีเนื้อหาซ้ำกัน 3 ย่อหน้า (เช่น บทสรุปท้ายบท):
```
ก่อน Dedup:  [หนังสือ A ย่อ 1] [หนังสือ A ย่อ 2] [หนังสือ A ย่อ 3] [หนังสือ B] [หนังสือ C] [หนังสือ D]
หลัง Dedup:  [หนังสือ A ย่อ 1] [หนังสือ B] [หนังสือ C] [หนังสือ D] [หนังสือ E] [หนังสือ F]
```
AI ได้ข้อมูลจาก **6 แหล่งที่หลากหลาย** แทนที่จะได้จาก 1 แหล่งซ้ำ 3 รอบ!

**Best Practice สำหรับ Deduplication:**
```python
# ✅ GOOD: ใช้ Set + prefix matching (O(1) lookup, เร็วมาก)
key = (book_title, content[:50])
if key not in seen: ...

# ❌ BAD: ใช้ nested loop เปรียบเทียบทั้ง string (O(n² × m) ช้ามาก)
for i in range(len(results)):
    for j in range(i+1, len(results)):
        if results[i]["content"] == results[j]["content"]: ...
```

### Step 5: Format Context (จัดรูปแบบสำหรับ LLM)
```python
# ตัวอย่าง context string ที่สุดท้ายจะถูกส่งให้ LLM:
"""
จากหนังสือ 'พลังแห่งความอดทน' กล่าวว่า:
\"\"\"
ความอดทนเป็นรากฐานของความสำเร็จ การฝึกฝนอย่างสม่ำเสมอ...
\"\"\"

---

จากหนังสือ 'จิตวิทยาผู้นำ' กล่าวว่า:
\"\"\"
ผู้นำที่ดีต้องมีความอดทนเป็นพิเศษ เพราะการเปลี่ยนแปลงองค์กร...
\"\"\"
"""
```

---

## 5. Relevance Threshold — ด่านตัดสินชี้ขาด

### อัลกอริทึม Routing (ตัดสินใจเลือกโหมด)
```python
RELEVANCE_THRESHOLD = 0.60
is_relevant = best_score >= RELEVANCE_THRESHOLD
```

### ตัวอย่างจริงจากการใช้งาน:
```
คำถาม: "หนังสือ 'พลังแห่งความอดทน' พูดถึงเรื่องอะไร?"
→ best_score = 0.78 → ≥ 0.60 ✅
→ 🟢 Curator Mode: ใช้ Context จากหนังสือ + Prompt พิเศษ

คำถาม: "วันนี้กินอะไรดี?"
→ best_score = 0.12 → < 0.60 ❌
→ 🔵 Conversational Mode: ตอบแบบแชทปกติ ไม่ยัด Context

คำถาม: "ความสำเร็จกับความอดทนเกี่ยวกันไหม?"
→ best_score = 0.63 → ≥ 0.60 ✅ (อยู่บนเส้น!)
→ 🟢 Curator Mode: แม้คะแนนไม่สูงมาก แต่ยังมี Context เสริมได้
```

### Best Practice สำหรับ Threshold:
```python
# ✅ GOOD: ตั้ง threshold พอเหมาะ (0.55-0.65)
RELEVANCE_THRESHOLD = 0.60  # เปิดช่องให้ค้นพบข้อมูลที่ "อาจเกี่ยวข้อง"

# ⚠️ ระวัง: ตั้งสูงเกินไป (0.85+)
RELEVANCE_THRESHOLD = 0.85  # แทบไม่เคยเปิด Curator Mode เลย!

# ⚠️ ระวัง: ตั้งต่ำเกินไป (0.30-)
RELEVANCE_THRESHOLD = 0.30  # ยัด Context ไม่เกี่ยวให้ AI ตลอด → คำตอบประหลาด
```

---

**→ ถัดไป:** ไฟล์ [04_deep_dive_learning_guide.md](./04_deep_dive_learning_guide.md) สำหรับการเรียนรู้เชิงลึก Trade-offs และ Scalability
