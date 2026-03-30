# 02 Language Model Core (สมองกลโมเดลภาษา)

เอกสารฉบับนี้เจาะลึกไปที่ `core/ai_core.py` ซึ่งเปรียบเสมือน "สมองส่วนกลาง" ของระบบ โดยเน้นเรื่องการนำโมเดลขนาดใหญ่มาทำงานบนเครื่องคอมพิวเตอร์แบบ Local และศิลปะการจัดการ Prompt

---

## 1. Local LLM & Quantization (การบีบอัดโมเดล)

โมเดลที่โปรเจกต์นี้ใช้คือ `openthaigpt1.5-7b-instruct` ซึ่งมีขนาดพารามิเตอร์ 7 พันล้านตัว (7B) 
ในสภาพปกติ การโหลดพารามิเตอร์ 7B ด้วยความแม่นยำระดับ 16-bit (bfloat16) จะต้องใช้ VRAM พื้นฐานราวๆ 14-15 GB ซึ่งหนักมากสำหรับการ์ดจอทั่วไป เช่น RTX 4060 (8GB VRAM)

**วิธีแก้ปัญหา: 4-bit NF4 Quantization**

เราใช้ `BitsAndBytesConfig` เข้ามาช่วย:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # NormalFloat 4-bit (ลดความสูญเสียข้อมูลที่สุด)
    bnb_4bit_compute_dtype=torch.bfloat16, # แต่ให้คณิตศาสตร์การบวก/คูณ ทำบนความแม่นยำ 16-bit
)
```
**การทำงานเบื้องหลัง (How it works):**
มันบีบข้อมูลน้ำหนัก (Weights) ของโมเดลให้เก็บในรูปแบบเลขความละเอียดต่ำเพียง 4-bit (ยัดลง VRAM ได้ที่ขนาดราวๆ 5-6 GB) แต่เมื่อถึงเวลาที่ต้องนำค่านั้นมา "คำนวณ" สร้างคำตอบ มันจะแตกค่าชั่วคราวเป็น 16-bit (`compute_dtype`) เรียกว่าการทำ "On-the-fly Dequantization" เพื่อคืนความฉลาดกลับมา

**ผลลัพธ์ทางสถาปัตยกรรม:** คุณใช้ทรัพยากรลดลงถึง 60% โดยแลกมากับการที่ความฉลาดของ AI ลดลงเพียงแค่ 1-2% เท่านั้น!

---

## 2. Universal Prompting: The Magic of `apply_chat_template`

เมื่อก่อน การส่งข้อความให้โมเดล AI ต้องพิมพ์แบบ Manual (เช่น นำหน้าคำด้วย `### Human:` หรือ `[INST]`) ทำให้เมื่อคุณอยากเปลี่ยนโมเดล (เช่น เปลี่ยนจาก Llama ไป Mistral) โปรแกรมจะพังเพราะ Prompt Format ไม่ตรงกัน

**การออกแบบใหม่:**
ระบบเก็บโครงสร้างบทสนทนาเป็นรายก้อน (Dictionary array) เรียกว่า Messages Array:
```python
messages = [
    {"role": "user", "content": "สวัสดี"},
    {"role": "assistant", "content": "สวัสดีครับ มีอะไรให้ช่วยไหม"},
    {"role": "user", "content": "ขอสูตรไข่เจียว"}
]
```
หลังจากนั้น ใช้วิธี `tokenizer.apply_chat_template()` เป็นคนประกอบร่างให้:
```python
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

**ทำไมวิธีนี้ถึงยอดเยี่ยม?**
แพ็คเกจ `tokenizer` ของ Hugging Face จะอ่านไฟล์ `tokenizer_config.json` ของโมเดลตัวนั้นๆ ហើយรับรู้ได้เองว่ามันควรจะแปลง Message array ให้ออกมาเป็น `[INST]...[/INST]` หรือ `<|im_start|>...<|im_end|>`
แปลว่า **โค้ดคุณจะทนทานต่อการเปลี่ยนแปลงในอนาคต 100% (Future-Proof)** แม้จะรันโมเดลค่ายเวทมนตร์ใดๆ โค้ดก็ไม่พัง

---

## 3. Short-Term Memory Management (หน่วยความจำระยะสั้นประหยัด VRAM)

ใน `ai_core.py` มีการออกแบบระบบ `_update_chat_history()`:
```python
max_messages = 4
if len(self.chat_history) > max_messages:
    self.chat_history = self.chat_history[-max_messages:]
```

**ทำไมถึงให้จำแค่ 4 ข้อความล่าสุด (2 Turns)?**
ในการรัน Local LLM, ขนาดของ "Context Window" ยิ่งยาวยิ่งกิน VRAM หนักเป็นทวีคูณ (ตามกฎ Quadratic Complexity ของ Transformer Attention) 
การตัดสินใจตัดความจำเหลือเพียง 2 บทสนทนาล่าสุด ถือเป็น **Proactive Optimization**:
1. **ลด VRAM OOM (Out of Memory):** ป้องกันการแครชเมื่อผู้ใช้คุยต่อเนื่องเป็นชั่วโมง
2. **ประหยัดค่าการประมวลผล (Token generation time):** ทุกครั้งที่พ่วงประวัติยาวเหยียด AI ต้องใช้เวลาอ่านนานขึ้น (Latency เพิ่ม)

---

**→ ขั้นตอนต่อไป:** ทำความเข้าใจกลไกการค้นหาและดึงความรู้ในไฟล์ `03_rag_engine.md`
