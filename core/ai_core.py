# core/ai_core.py (Migrated to OpenThaiGPT 2.0.0-Mistral)
# ---------------------------------------------------------
# Architecture:
#   - Uses Mistral Instruct format ([INST]...[/INST]) via apply_chat_template()
#   - Chat history stored as list of message dicts for proper multi-turn handling
#   - BitsAndBytes 4-bit quantization with NF4 + float16 compute dtype
# ---------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from core.rag_engine import BookRAGEngine
from core.config import MODEL_NAME
import re


class AICore:
    """
    Central AI engine that combines RAG retrieval with LLM generation.

    Responsibilities:
      - Load and manage the LLM (OpenThaiGPT 2.0.0-Mistral)
      - Load and manage the RAG knowledge base
      - Build prompts using Mistral's [INST] format via apply_chat_template()
      - Maintain short-term chat history for multi-turn conversations
    """

    def __init__(self):
        print("--- 🧠 Initializing AI Core ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.rag_engine = None
        self.system_prompt = self._get_system_prompt()
        self.curator_instruction = self._get_curator_instruction()
        # Chat history as list of message dicts for apply_chat_template()
        self.chat_history = []
        self._load_rag_engine()
        self._load_llm()
        print("--- ✅ AI Core is ready ---")

    # ------------------------------------------------------------------
    # Prompt Templates (adapted for Mistral Instruct format)
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """
        System-level instruction that defines the AI's persona.
        Used as the first message in every conversation.
        """
        return (
            "คุณคือ \"ภัณฑารักษ์ความรู้\" ผู้เชี่ยวชาญในการวิเคราะห์สังเคราะห์แนวคิดจากหนังสือ "
            "และสามารถสนทนาเชิงปัญญาได้อย่างเป็นธรรมชาติ ตอบเป็นภาษาไทยเสมอ"
        )

    def _get_curator_instruction(self) -> str:
        """
        Detailed instruction template for RAG-augmented (curator) mode.
        The {context} and {question} placeholders are filled at runtime.

        Design note: This is separated from the system prompt to keep
        the system prompt short and reusable across modes.
        """
        return """[เป้าหมายหลัก]
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
{question}"""

    # ------------------------------------------------------------------
    # Engine Loading
    # ------------------------------------------------------------------

    def _load_rag_engine(self):
        """Load the FAISS-based RAG knowledge base."""
        print("Loading Knowledge Base...")
        self.rag_engine = BookRAGEngine(index_path="data/index")
        print("✅ Knowledge Base is ready.")

    def _load_llm(self):
        """
        Load the LLM with 4-bit quantization.

        Key changes from v1.5:
          - Model: OpenThaiGPT 2.0.0-Mistral (Mistral architecture)
          - Quantization: NF4 with float16 compute dtype (was bfloat16)
          - No trust_remote_code needed for standard Mistral models
        """
        model_name = MODEL_NAME
        print(f"Loading Language Model '{model_name}' on {self.device}...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Mistral tokenizer may not have pad_token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("✅ Language Model is ready.")

    # ------------------------------------------------------------------
    # Chat History Management
    # ------------------------------------------------------------------

    def _update_chat_history(self, user_question: str, ai_answer: str):
        """
        Store conversation turns as message dicts for apply_chat_template().

        Keeps at most 2 previous turns (4 messages) to avoid exceeding
        the context window while maintaining conversational coherence.
        """
        self.chat_history.append({"role": "user", "content": user_question})
        self.chat_history.append({"role": "assistant", "content": ai_answer})

        # Keep only the last 2 turns (4 messages)
        max_messages = 4
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    # ------------------------------------------------------------------
    # Prompt Building (Mistral [INST] format via apply_chat_template)
    # ------------------------------------------------------------------

    def _build_messages(self, question: str, context: str = None, is_relevant: bool = False) -> list:
        """
        Build conversation messages list for apply_chat_template().

        Architecture decision: Using apply_chat_template() instead of
        manual [INST] formatting ensures compatibility across model
        versions and handles edge cases (BOS/EOS tokens) correctly.

        Args:
            question: The user's current question
            context: RAG context string (if relevant)
            is_relevant: Whether the RAG context is relevant enough to use

        Returns:
            List of message dicts in the format expected by apply_chat_template()
        """
        messages = []

        # Include chat history (previous turns)
        if self.chat_history:
            messages.extend(self.chat_history)

        # Build the current user message
        if is_relevant and context:
            # Curator mode: embed RAG context + curator instructions into user message
            sanitized_context = re.sub(r'[\(\[]\d+[\)\]]', '', context)
            sanitized_context = re.sub(r'\n+', '\n', sanitized_context).strip()
            user_content = self.curator_instruction.format(
                context=sanitized_context,
                question=question,
            )
        else:
            # Conversational mode: just the question
            user_content = question

        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Response Generation
    # ------------------------------------------------------------------

    def generate_response(self, question: str) -> dict:
        """
        Generate an AI response for the given question.

        Pipeline:
          1. Check for canned responses (greetings, etc.)
          2. RAG search for relevant context
          3. Build prompt using Mistral format via apply_chat_template()
          4. Generate tokens with sampling
          5. Clean up and return the answer

        Returns:
            dict with 'answer' (str) and 'sources' (list)
        """
        # --- Step 1: Canned responses for common greetings ---
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
            "ขอบคุณค่ะ": "ด้วยความยินดีครับ!",
        }
        if question.strip().lower() in canned_responses:
            return {"answer": canned_responses[question.strip().lower()], "sources": []}

        # --- Step 2: RAG search ---
        rag_results = self.rag_engine.search(query=question)
        context = rag_results["context"]
        sources = rag_results["sources"]
        best_score = rag_results["best_score"]

        RELEVANCE_THRESHOLD = 0.60
        is_relevant = best_score >= RELEVANCE_THRESHOLD

        if is_relevant:
            print(f"[AI_CORE] Relevant context found (Score: {best_score:.2f}). Using Curator mode.")
        else:
            print(f"[AI_CORE] Context not relevant (Score: {best_score:.2f}). Switching to conversational mode.")
            sources = []

        # --- Step 3: Build prompt using Mistral format ---
        messages = self._build_messages(
            question=question,
            context=context if is_relevant else None,
            is_relevant=is_relevant,
        )

        # apply_chat_template handles [INST]...[/INST] formatting automatically
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        # --- Step 4: Generate tokens ---
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # --- Step 5: Decode only the generated portion ---
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        clean_answer = raw_answer.strip()

        # --- Step 6: Update history and return ---
        self._update_chat_history(question, clean_answer)

        return {"answer": clean_answer, "sources": sources}