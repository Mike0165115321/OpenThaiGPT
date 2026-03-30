# 04 Deep Dive Learning Guide (คู่มือเจาะลึกสำหรับนักพัฒนา)

> 💡 **วัตถุประสงค์:** เอกสารฉบับนี้รวมทุกเรื่อง "เบื้องหลัง" ที่ไม่ได้อยู่ในโค้ดตรงๆ แต่เป็นเหตุผลว่า **"ทำไมต้องออกแบบแบบนี้"** เหมาะสำหรับผู้ที่ต้องการเข้าใจระบบอย่างลึกซึ้งเพื่อต่อยอดหรือปรับปรุง

---

## 1. Trade-Off Analysis: GPU VRAM Budget (งบประมาณหน่วยความจำ GPU)

### แผนผัง VRAM ของ GPU 8GB (RTX 4060):
```
┌──────────────────────────────────────────────────┐
│              GPU VRAM: 8,192 MB                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  ████████████████████████████  Model Weights      │
│  (4-bit Quantized)            ~5,200 MB (63%)    │
│                                                  │
│  ██████████████               KV Cache            │
│  (Context Window)             ~1,500 MB (18%)    │
│                                                  │
│  ████████                     Activations         │
│  (Compute Overhead)           ~800 MB (10%)      │
│                                                  │
│  ████                         CUDA Overhead       │
│  (Driver, Buffers)            ~400 MB (5%)       │
│                                                  │
│  ██                           Free Space          │
│                               ~292 MB (4%)       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### ข้อสรุปจากแผนผัง:
- **เหลือที่ว่างแค่ ~300 MB!** นี่คือเหตุผลว่าทำไม:
  1. **Embedder (bge-m3) ต้องอยู่บน CPU** — ถ้ายัดลง GPU จะ OOM ทันที
  2. **Chat History จำได้แค่ 2 turns** — ถ้าจำเยอะกว่านี้ KV Cache จะบวมจนล้น
  3. **ต้องใช้ 4-bit Quantization** — ถ้าใช้ 16-bit โมเดลกิน 14GB ส่วน GPU มี 8GB

---

## 2. Trade-Off: CPU Embedder vs GPU Embedder

### คำถามที่พบบ่อย:
*"ในเมื่อ GPU เร็วกว่า ทำไมถึงบังคับให้ BAAI-bge-m3 ทำงานบน CPU?"*

### การคำนวณเปรียบเทียบ:
```python
# rag_engine.py → __init__
self.device = "cpu"  # บังคับ CPU
self.embedder = SentenceTransformer(embedder_path, device=self.device)
```

| เกณฑ์ | Embedder บน GPU | Embedder บน CPU ✅ |
|-------|-----------------|-------------------|
| ความเร็ว Encode 1 query | ~50ms | ~300ms |
| VRAM ที่ใช้เพิ่ม | +1,500 MB 💀 | 0 MB ✅ |
| โมเดล LLM ยังรอดไหม? | ❌ OOM (5,200 + 1,500 = 6,700 + KV Cache = 💥) | ✅ รอดสบาย |
| ทำให้ผู้ใช้รอเพิ่ม | - | +250ms (แทบไม่รู้สึก!) |

### สรุป: แลก 250ms เพื่อความเสถียรของระบบ — **คุ้มค่ามาก**

### Best Practice:
```python
# ✅ BEST: แยก GPU สำหรับ LLM, CPU สำหรับ Embedder
embedder = SentenceTransformer(path, device="cpu")
llm = AutoModelForCausalLM.from_pretrained(..., device_map="auto")  # → GPU

# ❌ BAD: ยัดทุกอย่างลง GPU (จะ OOM)
embedder = SentenceTransformer(path, device="cuda")
llm = AutoModelForCausalLM.from_pretrained(..., device_map="auto")

# 🔮 IDEAL (ถ้ามี GPU 2 ตัว): แบ่งคนละ GPU
embedder = SentenceTransformer(path, device="cuda:0")
llm = AutoModelForCausalLM.from_pretrained(..., device_map={"": "cuda:1"})
```

---

## 3. Trade-Off: ทำไมถึงตัด Cross-Encoder Reranker ออก?

### RAG Pipeline แบบ Academic (ตำราเรียน):
```
Query → Retriever (FAISS) → Reranker (Cross-Encoder) → LLM
```

### RAG Pipeline ของเรา (Production-optimized):
```
Query → Retriever (FAISS) → LLM    ← ข้าม Reranker!
```

### เปรียบเทียบ:
| เกณฑ์ | มี Reranker | ไม่มี Reranker ✅ |
|-------|------------|-------------------|
| Response Time | 8-12 วินาที | 5-8 วินาที ✅ |
| VRAM เพิ่มเติม | +1-2 GB 💀 | 0 MB ✅ |
| คุณภาพการจัดลำดับ | ดีกว่า ~5-10% | ดีเพียงพอ ✅ |
| ความซับซ้อนโค้ด | เพิ่ม 1 class | Simple ✅ |

### เมื่อไหร่ควรเพิ่ม Reranker กลับมา?
```
✅ ควรเพิ่ม เมื่อ:
  - มี GPU VRAM เยอะ (24GB+)
  - ฐานข้อมูลใหญ่ (10,000+ documents)
  - ต้องการความแม่นยำในการจัดลำดับสูงมาก

❌ ยังไม่ต้อง เมื่อ:
  - GPU VRAM จำกัด (8GB) ← สถานการณ์ของเรา
  - ฐานข้อมูลเล็ก (หลักร้อย-พัน documents)
  - Embedder คุณภาพสูงอยู่แล้ว (bge-m3 เก่งพอ)
```

---

## 4. Deep Dive: Asynchronous Threading Strategy (main.py)

### ปัญหา: LLM Blocking คือ "ฆาตกรเงียบ" ของ Web Server

```python
# main.py
@app.post("/ask")
async def ask_question(question: Question):
    return await run_in_threadpool(ai_core.generate_response, question.text)
```

### ทำไมต้องใช้ `run_in_threadpool`?

**สถานการณ์จำลอง: 2 คนถามพร้อมกัน**

```
❌ ไม่ใช้ threadpool (Blocking):
เวลา →  0s ──────── 10s ──────── 20s
User A:  [████████████]              → ได้คำตอบ (10s)
User B:            [... รอ ...]  [████████████]  → ได้คำตอบ (20s!! 💀)
Server:  [BLOCKED ทั้ง server ไม่รับ request ใหม่]

✅ ใช้ threadpool (Non-blocking):
เวลา →  0s ──────── 10s
User A:  [████████████]  → ได้คำตอบ (10s)
User B:  [████████████]  → ได้คำตอบ (10s) ← ทำงานขนานกัน!
Server:  [ว่าง! พร้อมรับ request อื่น ✅]
```

### Best Practice:
```python
# ✅ GOOD: ย้าย blocking operation ไป thread pool
@app.post("/ask")
async def ask_question(question: Question):
    return await run_in_threadpool(ai_core.generate_response, question.text)

# ❌ BAD: เรียก blocking function ตรงๆ ใน async endpoint
@app.post("/ask")
async def ask_question(question: Question):
    return ai_core.generate_response(question.text)  # Server ค้างทั้งตัว!

# ❌ WORSE: ใช้ def แทน async def (FastAPI จะสร้าง thread ให้เอง แต่ไม่มีประสิทธิภาพ)
@app.post("/ask")
def ask_question(question: Question):
    return ai_core.generate_response(question.text)
```

---

## 5. Deep Dive: Prompt Engineering — Curator Mode

### ปัญหาของ LLM เมื่อได้รับ Context
เมื่อ AI ได้รับเนื้อหาจากหนังสือ มันมักจะ "ขี้เกียจ" และทำแค่ **Copy-Paste** เนื้อหากลับมา
หรือร้ายกว่านั้น — ใส่ Bullet Point, หัวข้อ, ตัวเลข จนอ่านเหมือนพนักงานค้นเอกสาร ไม่ใช่ผู้เชี่ยวชาญ

### วิธีแก้: Strict Formatting Constraints
```python
# ai_core.py → _get_curator_instruction()
"""
[กฎสำคัญที่สุด]
- ห้ามใช้หัวข้อกำกับใดๆ ทั้งสิ้นในคำตอบของคุณ
  (เช่น ห้ามพิมพ์คำว่า "ส่วนนำ:", "ส่วนวิเคราะห์:", "ส่วนสรุป:" ฯลฯ)
- ต้องเขียนให้อ่านลื่นไหลเหมือนบทความหรือการพูดคุย
"""
```

### ตัวอย่างผลลัพธ์:
```
❌ ไม่มี Strict Constraints (AI ตอบแบบพนักงาน):
"## สรุปจากหนังสือ
1. หนังสือ 'A' กล่าวว่า: ...
2. หนังสือ 'B' กล่าวว่า: ...
### สรุป
ดังนั้น ..."

✅ มี Strict Constraints (AI ตอบแบบผู้เชี่ยวชาญ):
"ความอดทนเป็นคุณสมบัติที่นักคิดหลายท่านให้ความสำคัญ
ในขณะที่หนังสือ 'พลังแห่งความอดทน' ให้มุมมองว่า
ความอดทนเป็นทักษะที่ฝึกได้ ไม่ใช่นิสัยติดตัว...
ในทางกลับกัน หนังสือ 'จิตวิทยาผู้นำ' กลับมองว่า...
คำถามที่น่าสนใจคือ แล้วคุณจะเลือกฝึกความอดทน
ด้วยวิธีไหน?"
```

### Best Practice สำหรับ Prompt Engineering:
```python
# ✅ GOOD: บอก AI ชัดเจนว่า "ห้ามทำอะไร" + "ให้ทำอะไร"
prompt = """
[กฎ]
- ห้ามใช้ Bullet Point
- ห้ามใช้หัวข้อ
- เขียนให้อ่านเหมือนบทความ
- อ้างอิงหนังสือด้วยสำนวนที่ลื่นไหล
"""

# ❌ BAD: บอกแค่ "ตอบให้ดี" (LLM ตีความไม่ได้)
prompt = "ตอบคำถามนี้ให้ดีที่สุด"

# ❌ BAD: ไม่มี constraints เลย (LLM จะ copy-paste)
prompt = f"นี่คือเนื้อหา: {context}\n\nคำถาม: {question}"
```

---

## 6. Scalability Blueprint (แผนขยายระบบ)

ถ้าวันนึงระบบนี้ต้องรองรับ 10,000 ผู้ใช้พร้อมกัน ต้องแก้คอขวดอะไรบ้าง?

### คอขวดที่ 1: LLM Generation (ร้ายแรงที่สุด)
```
ปัจจุบัน: GPU 1 ตัว → ประมวลผลทีละ 1 request → 10 วินาที/คน
ปัญหา:   10,000 คนรอคิว = 100,000 วินาที = 27 ชั่วโมง!

วิธีแก้ระดับ 1: ใช้ vLLM หรือ TGI (Text Generation Inference)
→ รวม requests เข้าเป็น Batch → ประมวลผลพร้อมกัน 8-32 requests
→ Throughput เพิ่ม 10-30 เท่า

วิธีแก้ระดับ 2: GPU หลายตัว + Load Balancer
→ GPU 4 ตัว × Batch 16 = 64 requests พร้อมกัน
→ รองรับ ~6,000 requests/นาที
```

### คอขวดที่ 2: FAISS In-Memory
```
ปัจจุบัน: FAISS อยู่ใน RAM → ข้อมูลมากขึ้น = กิน RAM มากขึ้น
ปัญหา:   1 ล้าน records × 1024 มิติ × 4 bytes = ~4 GB RAM

วิธีแก้: ย้ายไป Vector Database แยกเซิร์ฟเวอร์
→ Qdrant, Milvus, หรือ Weaviate
→ รองรับ Billion-scale vectors
→ มี Filtering, Sharding, Replication ในตัว
```

### คอขวดที่ 3: Session Management
```
ปัจจุบัน: chat_history อยู่ใน Python object → จำได้แค่ 1 คน!
ปัญหา:   User A กับ User B แชร์ history กัน 💀

วิธีแก้ ระดับ 1: เก็บ history ใน Dict ล็อกด้วย session_id
→ chat_histories = {"user_abc": [...], "user_xyz": [...]}

วิธีแก้ ระดับ 2 (Production): ย้ายไป Redis
→ redis.set(f"chat:{session_id}", json.dumps(history))
→ รองรับ 100,000+ sessions พร้อมกัน
→ มี TTL (auto-expire) ลบ sessions เก่าอัตโนมัติ
```

### สรุป Scalability Roadmap:
| ระดับ | ผู้ใช้ | สิ่งที่ต้องทำ |
|-------|--------|-------------|
| **Level 0** (ปัจจุบัน) | 1 คน | ไม่ต้องทำอะไร ✅ |
| **Level 1** | 10 คน | เพิ่ม session management (Dict) |
| **Level 2** | 100 คน | ใช้ vLLM/TGI + Redis |
| **Level 3** | 10,000+ | GPU Cluster + Vector DB + Load Balancer |

---

## 7. Resource Efficiency Summary (สรุปการ Optimize ทั้งหมดในโปรเจกต์)

| เทคนิค | ทรัพยากรที่ประหยัดได้ | Trade-off |
|--------|---------------------|-----------|
| 4-bit NF4 Quantization | VRAM ลด 60% (14→6 GB) | ความแม่นยำลด ~1-2% |
| Embedder บน CPU | VRAM ว่าง +1.5 GB | Latency เพิ่ม +250ms |
| ตัด Cross-Encoder | VRAM ว่าง +1-2 GB, เร็วขึ้น 3s | Ranking ลด ~5-10% |
| Chat History 2 turns | KV Cache ลด ~1.5 GB | จำบทสนทนาได้น้อย |
| Canned Responses | GPU ไม่ถูกใช้เลย (0 compute) | ตอบได้แค่คำทักทาย |
| FAISS FlatL2 | ไม่ต้อง build index ซับซ้อน | ค้นหาช้ากว่า HNSW (แต่ไม่สำคัญกับข้อมูลเล็ก) |
| `run_in_threadpool` | Event Loop ไม่ถูก block | ใช้ thread เพิ่ม (memory เล็กน้อย) |

---

*เมื่อคุณเข้าใจ "ทุกเหตุผล" เบื้องหลังการออกแบบ คุณก็พร้อมที่จะต่อยอด ปรับปรุง และ Scale ระบบนี้ไปสู่ระดับ Production ได้อย่างมั่นใจ!*

*หวังว่าเอกสารชุดนี้จะเป็นประโยชน์ต่อการเรียนรู้และพัฒนาด้าน AI Engineering ของคุณครับ* 🚀
