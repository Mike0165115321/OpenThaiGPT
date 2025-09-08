# main.py (The Correct Asynchronous Version)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from core.ai_core import AICore
from fastapi.concurrency import run_in_threadpool

app = FastAPI(title="AI API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="web/static"), name="static")

class Question(BaseModel): text: str
class Answer(BaseModel): answer: str; sources: list

ai_core = AICore()

@app.get("/")
def read_root(): return FileResponse("web/index.html")

@app.post("/new-chat")
def new_chat():
    print("[API] Clearing chat history.")
    ai_core.chat_history = []
    return {"status": "ok", "message": "Chat history cleared."}

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    return await run_in_threadpool(ai_core.generate_response, question.text)