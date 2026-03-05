# core/ai_core.py (Final Version with Safety Net)
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from core.rag_engine import BookRAGEngine
from core.config import MODEL_NAME

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids: list[int], device: str):
        self.stop_ids = torch.tensor(stop_ids).to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] >= self.stop_ids.shape[0]:
            if torch.equal(input_ids[0][-self.stop_ids.shape[0]:], self.stop_ids):
                return True
        return False

class AICore:
    def __init__(self):
        print("--- 🧠 Initializing AI Core ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None; self.tokenizer = None; self.rag_engine = None
        self.curator_prompt_template = self._get_curator_prompt_template()
        self.chat_history = []
        self._load_rag_engine()
        self._load_llm()
        print("--- ✅ AI Core is ready ---")

    def _get_curator_prompt_template(self): # Prompt v7.1 (Anti-header leakage)
        return """### Human:
[บทบาท]
คุณคือ "นักวิเคราะห์สังเคราะห์" ผู้เชี่ยวชาญในการกลั่นกรองแนวคิดจากหนังสือหลายเล่ม และสามารถสร้างบทสนทนาเชิงปัญญาได้อย่างเป็นธรรมชาติ

[เป้าหมายหลัก]
วิเคราะห์ "คำถาม" ของผู้ใช้ แล้วสร้างคำตอบที่ "ไร้รอยต่อ" โดยการผสานโครงสร้าง 3 ส่วนเข้าด้วยกันอย่างกลมกลืน (พิมพ์เฉพาะเนื้อหา ห้ามพิมพ์ชื่อส่วน)

[โครงสร้างคำตอบที่สมบูรณ์แบบ]
1. ส่วนนำ: เริ่มต้นด้วยการให้คำจำกัดความหรือมุมมองกว้างๆ เกี่ยวกับหัวข้อนั้นๆ เพื่อปูพื้นฐานการสนทนา
2. ส่วนวิเคราะห์: นำเสนอแนวคิดจากหนังสือแต่ละเล่มใน "บริบท" เป็นย่อหน้าแยกกัน เชื่อมโยงแต่ละย่อหน้าอย่างลื่นไหล ขึ้นต้นด้วยสำนวนเช่น "ในขณะที่หนังสือ '[ชื่อหนังสือ]' ให้มุมมองว่า..."
3. ส่วนสรุป: จบการวิเคราะห์ทั้งหมดด้วยย่อหน้าสุดท้ายที่เป็นคำถามปลายเปิด เพื่อกระตุ้นให้ผู้ใช้ไตร่ตรอง

[กฎสำคัญที่สุด]
- ห้ามใช้หัวข้อกำกับใดๆ ทั้งสิ้นในคำตอบของคุณ (เช่น ห้ามพิมพ์คำว่า "ส่วนนำ:", "ส่วนวิเคราะห์:", "ส่วนสรุป:", "เปิดประเด็นอย่างเป็นธรรมชาติ:", "คำถามชวนคิด:" ฯลฯ)
- ต้องเขียนให้อ่านลื่นไหลเหมือนบทความหรือการพูดคุย
- ยึดตามบริบท: อ้างอิงเฉพาะจากหนังสือที่อยู่ใน context
- ภาษา: ตอบเป็นภาษาไทยเท่านั้น

[บริบท]
{context}

[คำถาม]
{question}

### Assistant:
"""

    def _load_rag_engine(self):
        print("Loading Knowledge Base...")
        self.rag_engine = BookRAGEngine(index_path="data/index")
        print("✅ Knowledge Base is ready.")
    
    def _update_chat_history(self, user_question: str, ai_answer: str):
        self.chat_history.append(f"### Human:\n{user_question}")
        self.chat_history.append(f"### Assistant:\n{ai_answer}")
        
        max_history_items = 4
        if len(self.chat_history) > max_history_items:
            self.chat_history = self.chat_history[-max_history_items:]

    def _load_llm(self):
        model_name = MODEL_NAME
        print(f"Loading Language Model '{model_name}' on {self.device}...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        print("✅ Language Model is ready.")

    def generate_response(self, question: str) -> dict:
        
        canned_responses = {
            "สวัสดี": "สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ?",
            "สวัสดีครับ": "สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ?",
            "สวัสดีค่ะ": "สวัสดีครับ มีอะไรให้ผมช่วยไหมครับ?",
            "คุณชื่ออะไร": "ผมชื่อ OpenThaiGPT ครับ ยินดีที่ได้รู้จักครับ!",
            "คุณชื่ออะไรครับ": "ผมชื่อ OpenThaiGPT ครับ ยินดีที่ได้รู้จักครับ!",
            "คุณชื่ออะไรค่ะ": "ผมชื่อ OpenThaiGPT ครับ ยินดีที่ได้รู้จักครับ!",
            "คุณเป็นใคร": "ผมคือโมเดลภาษา OpenThaiGPT ครับ ถูกสร้างขึ้นเพื่อช่วยตอบคำถามและสนทนาในภาษาไทยครับ",
            "ขอบคุณ": "ด้วยความยินดีครับ!",
            "ขอบคุณครับ": "ด้วยความยินดีครับ!",
            "ขอบคุณค่ะ": "ด้วยความยินดีครับ!"
        }
        if question.strip().lower() in canned_responses:
            return {"answer": canned_responses[question.strip().lower()], "sources": []}

        rag_results = self.rag_engine.search(query=question)
        context, sources, best_score = rag_results["context"], rag_results["sources"], rag_results["best_score"]
        
        history_string = "\n".join(self.chat_history)
        
        RELEVANCE_THRESHOLD = 0.60
        is_relevant = best_score >= RELEVANCE_THRESHOLD
        
        if is_relevant:
            print(f"[AI_CORE] Relevant context found (Score: {best_score:.2f}). Using Curator mode.")
            sanitized_context = re.sub(r'[\(\[]\d+[\)\]]', '', context)
            sanitized_context = re.sub(r'\n+', '\n', sanitized_context).strip()
            prompt_body = self.curator_prompt_template.format(context=sanitized_context, question=question)
            final_prompt = f"{history_string}\n{prompt_body}" if self.chat_history else prompt_body
        else:
            print(f"[AI_CORE] Context not relevant (Score: {best_score:.2f}). Switching to conversational mode.")
            prompt_body = f"### Human:\n{question}\n\n### Assistant:"
            final_prompt = f"{history_string}\n{prompt_body}" if self.chat_history else prompt_body
            sources = []
        
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)

        stop_token_sequences = ["### Human:", "###"]
        stop_criterias = []
        for seq in stop_token_sequences:
            stop_ids = self.tokenizer.encode(seq, add_special_tokens=False)
            if stop_ids and stop_ids[0] == self.tokenizer.bos_token_id: stop_ids = stop_ids[1:]
            if stop_ids: stop_criterias.append(StopOnTokens(stop_ids, self.device))
        stopping_criteria = StoppingCriteriaList(stop_criterias)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        earliest_stop_pos = len(raw_answer)
        for seq in stop_token_sequences:
            pos = raw_answer.find(seq)
            if pos != -1:
                earliest_stop_pos = min(earliest_stop_pos, pos)
        
        clean_answer = raw_answer[:earliest_stop_pos].strip()

        self._update_chat_history(question, clean_answer)

        return {"answer": clean_answer, "sources": sources}