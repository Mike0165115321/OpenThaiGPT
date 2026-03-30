# 01 System Architecture (สถาปัตยกรรมภาพรวมของระบบ)

ยินดีต้อนรับสู่เอกสารสถาปัตยกรรมระบบ **OpenThaiGPT-RAG Chat**!
เอกสารฉบับนี้จะพาคุณทำความเข้าใจ "ภาพรวมทั้งหมด" ของระบบ ตั้งแต่ไฟล์แรกจนถึงไฟล์สุดท้าย — อธิบายว่าแต่ละชิ้นส่วนทำอะไร เชื่อมต่อกันอย่างไร และทำไมถึงออกแบบมาแบบนี้

---

## 1. What is OpenThaiGPT-RAG Chat? (โปรเจกต์นี้คืออะไร?)

โปรเจกต์นี้คือ **Web Application AI** ที่ทำหน้าที่เป็น "ภัณฑารักษ์ความรู้" (Knowledge Curator) ส่วนตัวของคุณ

**ปัญหาที่มันแก้:**
สมมติคุณมีหนังสือ 20 เล่ม เกี่ยวกับการพัฒนาตัวเอง คุณอยากจะถามว่า *"หนังสือทั้งหมดที่ผมมี สรุปว่าเราควรเริ่มต้นเปลี่ยนแปลงตัวเองอย่างไร?"*

ถ้าไม่มีระบบนี้ คุณต้องเปิดหนังสือทีละเล่ม อ่านทีละหน้า สรุปเอง
แต่ด้วยระบบ RAG ของเรา:
1. AI จะ **ค้นหา** เนื้อหาที่เกี่ยวข้องจากทุกเล่มอัตโนมัติ
2. AI จะ **สังเคราะห์** แนวคิดจากหลายเล่มมาร้อยเรียงเป็นคำตอบเดียว
3. AI จะ **อ้างอิง** ว่าข้อมูลมาจากหนังสือเล่มไหน

---

## 2. File Structure Map (แผนที่โครงสร้างไฟล์)

```
OpenThaiGPT/
├── main.py                  ← 🚪 ประตูหน้าบ้าน (API Gateway)
├── core/
│   ├── config.py            ← ⚙️  ศูนย์ควบคุมการตั้งค่า
│   ├── ai_core.py           ← 🧠 สมองกลาง (Orchestrator)
│   └── rag_engine.py        ← 📚 เครื่องยนต์ค้นหาความรู้
├── web/
│   ├── index.html           ← 🖥️  หน้าเว็บ UI
│   └── static/
│       └── script.js        ← 🎮 ตัวควบคุม UI (JavaScript)
├── data/
│   ├── source/              ← 📖 ไฟล์ .jsonl ต้นฉบับของหนังสือ
│   └── index/               ← 🗄️  ฐานข้อมูลเวกเตอร์ (FAISS)
├── build_index.py           ← 🏗️  สคริปต์สร้างฐานข้อมูล (รันครั้งเดียว)
├── download_model.py        ← ⬇️  สคริปต์ดาวน์โหลดโมเดล AI
└── requirements.txt         ← 📦 รายการ Dependencies
```

### คำอธิบายแต่ละไฟล์:

| ไฟล์ | หน้าที่ | เปรียบเทียบ |
|------|---------|-------------|
| `main.py` | รับ HTTP Request จาก Browser ส่งต่อให้ AI Core | เหมือน **พนักงานต้อนรับ** ที่ไม่รู้เรื่อง AI เลย แค่ส่งต่อคำถาม |
| `config.py` | เก็บค่าตั้งต้น (ชื่อโมเดล, path ข้อมูล) | เหมือน **สมุดโน้ตบนโต๊ะ** ที่เปลี่ยนแปลงได้ง่ายจุดเดียว |
| `ai_core.py` | โหลดโมเดล, สร้าง Prompt, สั่ง Generate | เหมือน **ผู้จัดการ** ที่ประสานงานระหว่าง RAG กับ LLM |
| `rag_engine.py` | ค้นหาเนื้อหาหนังสือที่เกี่ยวข้องกับคำถาม | เหมือน **บรรณารักษ์** ที่เก่งค้นหนังสือ |
| `build_index.py` | อ่านหนังสือทั้งหมดแล้วสร้าง Index | เหมือน **พนักงานจัดทำสารบัญ** (รันครั้งเดียว) |

---

## 3. System Component Diagram (แผนภาพส่วนประกอบ)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                  พิมพ์: "หนังสือ A สอนอะไร?"                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /ask
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API LAYER (main.py)                           │
│                                                                 │
│  @app.post("/ask")                                              │
│  async def ask_question(question):                              │
│      return await run_in_threadpool(                            │
│          ai_core.generate_response, question.text               │
│      )                                                          │
│                                                                 │
│  📝 หน้าที่: รับ Request → ส่งต่อ → ตอบ JSON กลับ                 │
│  ⚠️ ไม่รู้เรื่อง AI เลย! แค่เป็นตัวกลาง                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ function call
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AI CORE (ai_core.py)                             │
│                                                                 │
│  generate_response("หนังสือ A สอนอะไร?")                         │
│                                                                 │
│  Step 1: เช็ค canned response → ไม่ใช่คำทักทาย → ไปต่อ            │
│  Step 2: ส่งคำถามไป RAG Engine ────────────────────┐             │
│  Step 3: ได้ context + score กลับมา ◄──────────────┘             │
│  Step 4: score ≥ 0.60? → ใช่! → เปิดโหมดภัณฑารักษ์               │
│  Step 5: สร้าง Prompt + ใส่ context + chat history               │
│  Step 6: ส่งให้ LLM generate ─────────────────────┐             │
│  Step 7: ได้คำตอบ ◄──────────────────────────────┘              │
│  Step 8: ตัดแต่งข้อความ → return JSON                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│   RAG ENGINE         │  │   LANGUAGE MODEL (LLM)               │
│   (rag_engine.py)    │  │   (OpenThaiGPT 1.5-7B)               │
│                      │  │                                      │
│                      │  │   🔥 ทำงานบน GPU (4-bit quantized)   │
│  📚 ค้นหาหนังสือ      │  │   📝 สร้างข้อความตอบ                   │
│  🧠 Embedder: bge-m3 │  │   🧠 เข้าใจภาษาไทย                    │
│  🗄️ FAISS Index      │  │   💬 จำบทสนทนาล่าสุด                  │
│                      │  │                                      │
│  ⚡ ทำงานบน CPU       │  │                                      │
└──────────────────────┘  └──────────────────────────────────────┘
```

---

## 4. The Request Lifecycle (เส้นทางชีวิตของ 1 คำถาม)

มาดูกันทีละขั้นตอนว่าเกิดอะไรขึ้นเมื่อผู้ใช้ถาม *"ความกตัญญูคืออะไร?"*

### ขั้นตอนที่ 1: Frontend → API (การส่งคำถาม)
```javascript
// web/static/script.js
const response = await fetch('/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text: "ความกตัญญูคืออะไร?"})
});
```
JavaScript ส่ง HTTP POST ไปที่ `/ask` พร้อม payload `{"text": "ความกตัญญูคืออะไร?"}`

### ขั้นตอนที่ 2: API Gateway (main.py รับงาน)
```python
# main.py
@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    return await run_in_threadpool(ai_core.generate_response, question.text)
```
**จุดสำคัญ:** `run_in_threadpool()` — ไม่ได้เรียก `generate_response()` ตรงๆ แต่โยนไปทำงานใน Thread แยก
- **ทำไม?** เพราะ `model.generate()` เป็น blocking operation (ใช้เวลา 5-30 วินาที)
- **ถ้าไม่ใช้ threadpool?** เว็บจะค้างทั้งเซิร์ฟเวอร์ ไม่สามารถรับ request อื่นได้เลยจนกว่า AI จะตอบเสร็จ

### ขั้นตอนที่ 3: Canned Response Check (เช็คคำทักทาย)
```python
# ai_core.py → generate_response()
canned_responses = {
    "สวัสดี": "สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ?",
    "ขอบคุณ": "ด้วยความยินดีครับ!",
    ...
}
if question.strip().lower() in canned_responses:
    return {"answer": canned_responses[...], "sources": []}  # ⚡ O(1) ตอบทันที
```
**ทำไมต้องมี?** ถ้าคนทักทายว่า "สวัสดี" แล้วต้องรอ AI ประมวลผล 10 วินาที = ประสบการณ์ผู้ใช้แย่มาก
**Best Practice:** ใส่ canned response สำหรับคำที่ใช้บ่อยที่สุด เพื่อตอบ < 1ms โดยไม่ต้องเปลือง GPU เลย

### ขั้นตอนที่ 4: RAG Search (ค้นหาความรู้)
```python
rag_results = self.rag_engine.search(query="ความกตัญญูคืออะไร?")
# ผลลัพธ์:
# {
#     "context": "จากหนังสือ 'คุณธรรมในสังคมไทย' กล่าวว่า: ...",
#     "sources": ["คุณธรรมในสังคมไทย", "จริยธรรมเพื่อชีวิต"],
#     "best_score": 0.72
# }
```
RAG Engine แปลงคำถามเป็น Vector แล้วค้นหาเนื้อหาจากหนังสือที่ใกล้เคียงที่สุด

### ขั้นตอนที่ 5: Relevance Routing (ตัดสินใจเลือกโหมด)
```python
RELEVANCE_THRESHOLD = 0.60
is_relevant = best_score >= RELEVANCE_THRESHOLD  # 0.72 >= 0.60 → True!
```

**ตัวอย่างเปรียบเทียบ:**
| คำถาม | best_score | โหมดที่เลือก |
|--------|-----------|-------------|
| "ความกตัญญูคืออะไร?" | 0.72 | 🟢 Curator Mode (ใช้หนังสือ) |
| "วันนี้อากาศเป็นไง?" | 0.15 | 🔵 Conversational Mode (แชทปกติ) |
| "สวัสดีครับ" | — | ⚡ Canned Response (ตอบทันที) |

### ขั้นตอนที่ 6: Prompt Building + LLM Generation
```python
# สร้าง Messages array → แปลงเป็น Prompt → ส่งให้โมเดล
messages = self._build_messages(question, context, is_relevant=True)
prompt_text = self.tokenizer.apply_chat_template(messages, ...)
outputs = self.model.generate(**inputs, max_new_tokens=1024, ...)
```

### ขั้นตอนที่ 7: Response (ส่งคำตอบกลับ)
```json
{
    "answer": "ความกตัญญูเป็นหนึ่งในคุณธรรมสำคัญที่...",
    "sources": ["คุณธรรมในสังคมไทย", "จริยธรรมเพื่อชีวิต"]
}
```

---

## 5. Separation of Concerns (SoC) — หลักการแยกส่วนความรับผิดชอบ

การแยก SoC ในโปรเจกต์นี้มีเป้าหมาย: **"ถ้าจะเปลี่ยนส่วนใดส่วนหนึ่ง ส่วนอื่นจะไม่พัง"**

### ตัวอย่างเชิงปฏิบัติ:

| สถานการณ์ | ไฟล์ที่ต้องแก้ | ไฟล์ที่ไม่ต้องแตะเลย |
|-----------|---------------|---------------------|
| เปลี่ยนโมเดลจาก OpenThaiGPT → Typhoon | `config.py` เท่านั้น! | `main.py`, `rag_engine.py`, `web/` |
| เปลี่ยนจากเว็บเป็น Line Bot | `main.py` | `ai_core.py`, `rag_engine.py` |
| เปลี่ยนจาก FAISS → Qdrant Cloud | `rag_engine.py`, `build_index.py` | `ai_core.py`, `main.py` |
| เพิ่มหนังสือเล่มใหม่ | รัน `build_index.py` ใหม่ | ไม่ต้องแก้โค้ดเลย! |

### Best Practice: ทำไมต้องแยก `config.py` ออกมา?
```python
# ❌ BAD: Hardcode ชื่อโมเดลกระจายทั่วโค้ด
model = AutoModelForCausalLM.from_pretrained("openthaigpt/openthaigpt1.5-7b-instruct")

# ✅ GOOD: ดึงจาก config จุดเดียว
from core.config import MODEL_NAME
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
```
**เหตุผล:** ถ้าวันนึงต้องเปลี่ยนโมเดล คุณแก้แค่ 1 บรรทัดใน `config.py` แค่นั้น ไม่ต้องไล่หาทั่วโปรเจกต์

---

## 6. Best Practices ที่ใช้ในโปรเจกต์นี้

### 6.1 Single Responsibility Principle (SRP)
ทุกไฟล์มี "งานเดียว" ที่ชัดเจน:
- `main.py` → รับ/ส่ง HTTP เท่านั้น (ไม่มี business logic)
- `ai_core.py` → ประสานงาน AI เท่านั้น (ไม่มี DB access)
- `rag_engine.py` → ค้นหาข้อมูลเท่านั้น (ไม่รู้จักโมเดลภาษา)

### 6.2 Abstraction Layers
`ai_core.py` ไม่จำเป็นต้องรู้ว่า RAG ใช้ FAISS หรือ Qdrant — มันแค่เรียก `self.rag_engine.search(query)` แล้วรอผลลัพธ์ ถ้าวันนึงเปลี่ยน Database ภายใน `rag_engine.py` ตัว `ai_core.py` ไม่มีทางรู้เลย (Loose Coupling)

### 6.3 Fail Gracefully
```python
# rag_engine.py
if not self.index or not self.mapping:
    return {"context": "Error: Knowledge base is not loaded.", "sources": [], "best_score": 0.0}
```
**จุดสำคัญ:** ถ้า FAISS โหลดไม่สำเร็จ ระบบ AI ยังทำงานได้ — แค่ตอบแบบไม่มี Context (Graceful Degradation) ไม่ crash ทั้ง app!

---

**→ ถัดไป:** เจาะลึกสมอง AI (LLM) ที่ไฟล์ [02_language_model.md](./02_language_model.md)
