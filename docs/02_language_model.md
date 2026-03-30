# 02 Language Model Core (สมองกลโมเดลภาษา)

เอกสารฉบับนี้เจาะลึกที่ `core/ai_core.py` ซึ่งเปรียบเสมือน "สมองส่วนกลาง" ของระบบ
เราจะทำความเข้าใจว่าโมเดลภาษาขนาดใหญ่ (LLM) ถูกโหลดและใช้งานอย่างไรบนเครื่อง Local ที่มีทรัพยากรจำกัด

---

## 1. LLM คืออะไร? (ปูพื้นฐาน)

**Large Language Model (LLM)** คือโมเดล AI ขนาดใหญ่ที่ถูกเทรนด้วยข้อมูลเป็นล้านๆ ข้อความ เพื่อให้มัน "เข้าใจ" และ "สร้าง" ภาษาได้

โมเดลที่เราใช้: `openthaigpt1.5-7b-instruct`
- **7B** = มีพารามิเตอร์ 7,000,000,000 ตัว (7 พันล้าน!)
- **instruct** = ถูกฝึกมาให้ "ทำตามคำสั่ง" ได้ (ไม่ใช่แค่เติมข้อความ)

### เปรียบเทียบให้เห็นภาพ:
| แนวคิด | เปรียบเทียบ |
|--------|-------------|
| พารามิเตอร์ | เหมือน "เซลล์ประสาท" ในสมอง ยิ่งเยอะยิ่งฉลาด |
| Weights (น้ำหนัก) | เหมือน "ความแข็งแรงของการเชื่อมต่อ" ระหว่างเซลล์ |
| Token | หน่วยย่อยของข้อความ (คำหนึ่งอาจมี 1-3 tokens) |
| Context Window | เหมือน "สมุดกระดาษ" ที่ AI อ่านได้ครั้งละกี่หน้า |

---

## 2. Quantization คืออะไร? (การบีบอัดโมเดลให้รันได้บนเครื่องจริง)

### ปัญหา: โมเดล 7B ใหญ่เกินไป
ทุกพารามิเตอร์ถูกเก็บเป็นตัวเลข ถ้าเก็บแบบ **16-bit** (ความแม่นยำปกติ):
```
7,000,000,000 × 2 bytes (16-bit) = 14 GB VRAM
```
การ์ดจอ RTX 4060 มี VRAM เพียง **8 GB** → โมเดลใส่ไม่ลง!

### วิธีแก้: 4-bit NF4 Quantization
บีบอัดพารามิเตอร์จาก 16-bit เหลือ 4-bit:
```
7,000,000,000 × 0.5 bytes (4-bit) = 3.5 GB VRAM (+ overhead ≈ 5-6 GB)
```
ใส่การ์ดจอ 8 GB ได้สบายๆ!

### โค้ดที่ใช้จริง:
```python
# ai_core.py → _load_llm()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # ① เก็บน้ำหนักแบบ 4-bit
    bnb_4bit_quant_type="nf4",            # ② ใช้อัลกอริทึม NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16 # ③ แต่ "คำนวณ" ด้วย 16-bit
)
```

### แต่ละบรรทัดทำอะไร?

**① `load_in_4bit=True`**
สั่งให้ Hugging Face โหลดน้ำหนักโมเดลในรูปแบบ 4-bit แทน 16-bit

**② `bnb_4bit_quant_type="nf4"`**
เลือกอัลกอริทึมการบีบอัดแบบ **NormalFloat 4-bit (NF4)** 
- ทำไมไม่ใช้ `int4` ธรรมดา? เพราะ NF4 ถูกออกแบบมาให้กระจายค่าตัวเลขตามการกระจายแบบ Normal Distribution ซึ่งตรงกับลักษณะน้ำหนักของ Neural Network พอดี
- **ผลลัพธ์:** สูญเสียความแม่นยำน้อยกว่า int4 ถึง 30-40%

**③ `bnb_4bit_compute_dtype=torch.bfloat16`**
- น้ำหนักถูก **"เก็บ"** เป็น 4-bit (ประหยัดพื้นที่)
- แต่ตอน **"คำนวณ"** จะถูกขยายเป็น 16-bit ชั่วคราว (รักษาความแม่นยำ)
- เรียกว่า "On-the-fly Dequantization"

### ตัวอย่างเปรียบเทียบ:
```
📦 เก็บของ (Storage): ใส่กล่องเล็ก 4-bit → ประหยัดพื้นที่ตู้ (VRAM)
🔬 ใช้งาน (Compute): เอาออกมาแกะ → กางออกเป็น 16-bit → ทำงานแม่นยำ
📦 เก็บกลับ: ยัดกลับเข้ากล่อง 4-bit
```

### Best Practice สำหรับ Quantization:
```python
# ✅ BEST: NF4 + bfloat16 (แม่นยำที่สุด, ประหยัดที่สุด)
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16)

# 🟡 OK: NF4 + float16 (สำหรับ GPU เก่าที่ไม่รองรับ bfloat16)
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.float16)

# ❌ BAD: ไม่ใช้ Quantization เลย (VRAM ไม่พอ, โมเดลจะ crash)
model = AutoModelForCausalLM.from_pretrained(model_name)  # กิน 14 GB!
```

---

## 3. Universal Prompting ด้วย `apply_chat_template()`

### ปัญหาเดิม: Prompt Format ผูกติดกับโมเดล
โมเดลแต่ละตัวมี "ภาษาถิ่น" ต่างกัน:
```
# OpenThaiGPT 1.5 (Qwen-based) ใช้:
<|im_start|>user
สวัสดีครับ<|im_end|>
<|im_start|>assistant

# Mistral ใช้:
[INST] สวัสดีครับ [/INST]

# Llama 3 ใช้:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
สวัสดีครับ<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

ถ้า hardcode format ไว้ในโค้ด เมื่อเปลี่ยนโมเดล → ต้องแก้โค้ดทั้งหมด!

### วิธีแก้: Universal Messages Array
เราเก็บบทสนทนาเป็น **รูปแบบกลาง (Intermediate Format)** ที่ทุกโมเดลเข้าใจ:
```python
messages = [
    {"role": "user", "content": "ความสุขคืออะไร?"},
    {"role": "assistant", "content": "ความสุขคือสถานะของจิตใจที่..."},
    {"role": "user", "content": "แล้วหนังสือเล่มไหนพูดถึงเรื่องนี้?"}
]
```

จากนั้น `apply_chat_template()` จะแปลงให้อัตโนมัติ:
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # คืนค่าเป็น string (ยังไม่แตก tokens)
    add_generation_prompt=True  # เพิ่มท่อนเปิดทางให้ AI เริ่มตอบ
)
```

**ผลลัพธ์:** ถ้ารันกับ OpenThaiGPT 1.5 (Qwen-based) จะได้:
```
<|im_start|>user
ความสุขคืออะไร?<|im_end|>
<|im_start|>assistant
ความสุขคือสถานะของจิตใจที่...<|im_end|>
<|im_start|>user
แล้วหนังสือเล่มไหนพูดถึงเรื่องนี้?<|im_end|>
<|im_start|>assistant
```

**Best Practice สำหรับ Prompting:**
```python
# ✅ GOOD: ใช้ messages + apply_chat_template (Universal, Future-proof)
messages = [{"role": "user", "content": question}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ❌ BAD: Hardcode format (พังทันทีเมื่อเปลี่ยนโมเดล)
prompt = f"### Human:\n{question}\n\n### Assistant:\n"
```

---

## 4. Chat History Management (ระบบความจำระยะสั้น)

### ทำงานอย่างไร?
```python
def _update_chat_history(self, user_question: str, ai_answer: str):
    self.chat_history.append({"role": "user", "content": user_question})
    self.chat_history.append({"role": "assistant", "content": ai_answer})

    # เก็บแค่ 4 ข้อความล่าสุด (2 turns: 1 turn = คำถาม + คำตอบ)
    max_messages = 4
    if len(self.chat_history) > max_messages:
        self.chat_history = self.chat_history[-max_messages:]
```

### ทำไมต้องจำกัดแค่ 2 turns?

**เหตุผลทางเทคนิค: Transformer Attention Complexity**
ความซับซ้อนของ Transformer อยู่ที่ **O(n²)** — ถ้าจำ context ยาวขึ้น 2 เท่า ใช้ VRAM มากขึ้น 4 เท่า!

```
ตัวอย่างจำลอง:
┌──────────────────────────────┬──────────┬─────────────────┐
│ จำนวน Turns ที่จำ             │ ~Tokens  │ VRAM ใช้เพิ่ม     │
├──────────────────────────────┼──────────┼─────────────────┤
│ 2 turns (4 messages) ✅      │ ~500     │ +200 MB         │
│ 5 turns (10 messages) ⚠️    │ ~1,200   │ +800 MB         │
│ 10 turns (20 messages) ❌    │ ~2,500   │ +2 GB (OOM!)    │
└──────────────────────────────┴──────────┴─────────────────┘
```
บน GPU 8GB ที่โมเดลกิน 6GB อยู่แล้ว → เหลือ headroom แค่ ~2GB → จำได้แค่ 2 turns อย่างปลอดภัย

### Best Practice สำหรับ Chat History:
```python
# ✅ GOOD: ตัดเก่าออกแบบ sliding window (เก็บเฉพาะล่าสุด)
self.chat_history = self.chat_history[-max_messages:]

# ❌ BAD: เก็บทั้งหมดจนกว่า session จะจบ (OOM แน่นอน)
self.chat_history.append(message)  # ไม่มีการตัด!

# 🔮 FUTURE (ถ้ามี VRAM มากพอ): ใช้ Summarization
# สรุปประวัติเก่า → เก็บเป็น 1 ข้อความ → ประหยัด tokens
```

---

## 5. Token Generation Parameters (พารามิเตอร์การสร้างข้อความ)

```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=1024,     # สร้างได้สูงสุด 1024 tokens (~500 คำไทย)
    do_sample=True,          # เปิดการสุ่ม (ไม่เอาคำตอบเดิมซ้ำๆ)
    temperature=0.6,         # ③ ระดับความสร้างสรรค์
    top_p=0.95,              # ④ สุ่มจากกลุ่มคำที่น่าจะถูกต้อง 95%
    repetition_penalty=1.1,  # ⑤ ลงโทษ AI ที่พูดซ้ำ
    pad_token_id=...,
    eos_token_id=...,
)
```

### อธิบายแต่ละตัว:

**③ `temperature=0.6` (ระดับความสร้างสรรค์)**
```
temperature = 0.1  → AI เลือกคำที่ "ปลอดภัยที่สุด" เสมอ (น่าเบื่อ, วิชาการมาก)
temperature = 0.6  → สมดุลระหว่างความถูกต้องกับความน่าอ่าน ✅ (ค่าที่เราใช้)
temperature = 1.5  → AI จะ "ฝันเฟื่อง" สร้างคำแปลกๆ เยอะ (Hallucination สูง)
```

**④ `top_p=0.95` (Nucleus Sampling)**
แทนที่จะสุ่มจากคำทุกคำในพจนานุกรม ให้สุ่มเฉพาะจากกลุ่มคำที่มีความน่าจะเป็นรวมกัน 95%
```
สมมติโมเดลเราต้องเลือกคำถัดไปหลังจาก "ฉันชอบกิน":
  ข้าว (40%), ก๋วยเตี๋ยว (30%), ส้มตำ (15%), มะม่วง (10%)
  → top_p=0.95 จะสุ่มจาก 4 คำนี้ (รวม = 95%)
  แทนที่จะเสี่ยงไปสุ่ม "รถยนต์" (0.01%) ที่ไม่เกี่ยวเลย
```

**⑤ `repetition_penalty=1.1`**
ถ้า AI พูดคำไหนไปแล้ว ความน่าจะเป็นของคำนั้นจะถูก "ลงโทษ" หารด้วย 1.1
ป้องกันอาการ: *"ความสุขคือความสุข ความสุขเป็นสิ่งที่ทำให้เรามีความสุข ความสุข..."*

---

**→ ถัดไป:** เจาะลึกระบบค้นหาความรู้ (RAG) ที่ไฟล์ [03_rag_engine.md](./03_rag_engine.md)
