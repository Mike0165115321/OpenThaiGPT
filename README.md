# OpenThaiGPT-RAG Chat: ภัณฑารักษ์ความรู้ส่วนตัวของคุณ

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30+-yellow?style=for-the-badge)](https://huggingface.co/docs/transformers/index)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.3+-blue?style=for-the-badge&logo=tailwindcss)](https://tailwindcss.com/)

**OpenThaiGPT-RAG Chat** คือ Web Application AI อัจฉริยะที่ทำหน้าที่เป็น "ภัณฑารักษ์ความรู้" (Knowledge Curator) ส่วนตัวของคุณ โปรเจคนี้ใช้สถาปัตยกรรม Retrieval-Augmented Generation (RAG) เพื่อตอบคำถามจากคลังความรู้ (หนังสือ) ที่คุณกำหนดเอง โดยมี `openthaigpt-1.5-7b-instruct` เป็นสมองหลักในการสังเคราะห์คำตอบ

โปรเจคนี้ถูกออกแบบมาให้สามารถสนทนาได้อย่างชาญฉลาด โดยสามารถปรับเปลี่ยนพฤติกรรมได้ตามความเหมาะสม:
*   **โหมดภัณฑารักษ์ความรู้:** เมื่อคำถามมีความเกี่ยวข้องกับเนื้อหาในหนังสือ AI จะสวมบทบาทเป็น "นักวิเคราะห์สังเคราะห์" ที่สามารถร้อยเรียงแนวคิดจากหนังสือหลายเล่มให้กลายเป็นบทวิเคราะห์ที่เฉียบคมและเป็นธรรมชาติ
*   **โหมดสนทนาทั่วไป:** เมื่อคำถามเป็นการทักทาย, ไม่ชัดเจน, หรือไม่เกี่ยวข้องกับเนื้อหาในคลังความรู้ AI จะตอบกลับอย่างเป็นธรรมชาติและรู้ว่าเมื่อไหร่ควรจะบอกว่า "ไม่เข้าใจ"

![Screenshot of the application](https://i.imgur.com/your-screenshot-url.png) 
*(แนะนำ: ให้คุณถ่ายภาพหน้าจอสวยๆ ของโปรเจค แล้วอัปโหลดขึ้น [Imgur](https://imgur.com/) หรือบริการฝากรูปอื่นๆ แล้วนำลิงก์มาใส่แทนที่ตรงนี้)*

---

## 🚀 คุณสมบัติเด่น (Features)

*   **🧠 สมอง AI ทรงพลัง:** ขับเคลื่อนด้วย `openthaigpt-1.5-7b-instruct` พร้อม 4-bit Quantization เพื่อประสิทธิภาพสูงสุดบน GPU ระดับ Consumer
*   **📚 ระบบ RAG อัจฉริยะ:** ใช้ `sentence-transformers` และ `faiss` ในการค้นหาและจัดลำดับความเกี่ยวข้องของข้อมูลจากคลังความรู้ของคุณ
*   **🤖 AI ที่ปรับตัวได้:** มีระบบคัดกรอง (Triage System) เพื่อเลือกว่าจะตอบในโหมด "ภัณฑารักษ์" หรือโหมด "สนทนาทั่วไป"
*   **🧠 ความจำระยะสั้น:** สามารถจดจำบริบทของบทสนทนาก่อนหน้าเพื่อการพูดคุยที่ลื่นไหล
*   **🌐 API Backend:** สร้างด้วย FastAPI ที่แข็งแกร่งและเสถียร
*   **✨ UI ทันสมัย:** Frontend ที่สวยงามและใช้งานง่าย สร้างด้วย HTML และ Tailwind CSS (ผ่าน CDN)

---

## 🛠️ การติดตั้ง (Installation)

โปรเจคนี้ถูกพัฒนาและทดสอบบน WSL (Ubuntu)

**1. Clone Repository:**
```bash
git clone https://github.com/Mike0165115321/OpenThaiGPT.git
cd OpenThaiGPT
```

**2. สร้างและเปิดใช้งาน Virtual Environment:**
```bash
python3 -m venv otgpt-env
source otgpt-env/bin/activate
```

**3. ติดตั้ง Dependencies:**
```bash
pip install -r requirements.txt
```

**4. ดาวน์โหลดโมเดลภาษา (LLM):**
สคริปต์นี้จะดาวน์โหลดโมเดล `openthaigpt-1.5-7b-instruct` (ขนาดประมาณ 15 GB) มาไว้ในโฟลเดอร์ `models/`
```bash
python download_model.py
```

---

## 📖 การเตรียมข้อมูล (Data Preparation)

**1. สร้างไฟล์ข้อมูล `.jsonl`:**
โปรเจคนี้อ่านข้อมูลจากไฟล์ `.jsonl` ที่อยู่ในโฟลเดอร์ `data/source/` โดยแต่ละบรรทัดคือหนึ่ง "เอกสาร" (passage) ในรูปแบบ JSON

**Format ที่ต้องการ:**
```json
{"book_title": "ชื่อหนังสือของคุณ", "content": "เนื้อหาของส่วนนี้..."}
{"book_title": "ชื่อหนังสือของคุณ", "content": "เนื้อหาของส่วนถัดไป..."}
```

คุณสามารถใช้เครื่องมือสำหรับประมวลผลหนังสือได้จากโปรเจคนี้: **[ลิงก์ไปยังโปรเจค Book Processor ของคุณ]** 
*(สำคัญ: แก้ไขลิงก์นี้ให้ถูกต้อง)*

**2. สร้าง Index สำหรับการค้นหา:**
หลังจากนำไฟล์ `.jsonl` ทั้งหมดมาใส่ใน `data/source/` แล้ว ให้รันสคริปต์นี้เพื่อสร้าง FAISS index:
```bash
python build_index.py```
สคริปต์นี้จะสร้างไฟล์ `faiss_index.bin` และ `faiss_mapping.json` ในโฟลเดอร์ `data/index/`

---

## ▶️ การใช้งาน (Usage)

เมื่อติดตั้งและเตรียมข้อมูลเสร็จเรียบร้อยแล้ว ให้รัน API server:
```bash
uvicorn main:app --reload --port 8003
```

จากนั้นเปิดเว็บเบราว์เซอร์ของคุณแล้วไปที่:
**`http://127.0.0.1:8003`**

คุณสามารถเริ่มสนทนากับ "ภัณฑารักษ์ความรู้" ของคุณได้ทันที!

---

## 💡 แนวทางการพัฒนาต่อ (Future Work)

*   [ ] **Chat History:** ทำให้ Sidebar แสดงประวัติการสนทนาและสามารถกดกลับไปดูได้
*   [ ] **Streaming Response:** กลับไปท้าทายการทำให้ AI ตอบกลับแบบ Streaming อีกครั้ง
*   [ ] **Knowledge Graph:** เพิ่มความสามารถในการทำความเข้าใจความสัมพันธ์ของข้อมูลด้วย Knowledge Graph

---

## 🙏 ขอขอบคุณ

*   **OpenThaiGPT:** สำหรับโมเดลภาษาไทยที่ทรงพลัง
*   **Hugging Face:** สำหรับ Ecosystem ที่ยอดเยี่ยม
*   และผู้สร้างไลบรารีโอเพนซอร์สทั้งหมดที่ทำให้โปรเจคนี้เป็นไปได้
