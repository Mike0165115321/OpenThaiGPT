# build_index.py
# Adapted from PROJECT NEXUS (V4.2 - Hotfix & Efficient RAG Architect)

import os
import json
import faiss
from sentence_transformers import SentenceTransformer
import torch
import re
from typing import List, Dict

class RAGIndexBuilder:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  RAG Index Builder initializing on device: {device.upper()}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"✅ Embedding model '{model_name}' loaded successfully.")

    def load_all_data(self, data_folder: str) -> List[Dict]:
        """
        ปรับปรุงใหม่: โหลดข้อมูลทั้งหมดจากไฟล์ .jsonl ทุกไฟล์มารวมกันเป็น List เดียว
        """
        all_items = []
        print(f"\n--- 📚 Loading all book data from '{data_folder}' ---")
        
        files_to_process = sorted([f for f in os.listdir(data_folder) if f.endswith(".jsonl")])
        
        if not files_to_process:
            print("  - 🟡 No .jsonl files found to process.")
            return []

        print(f"  - Found {len(files_to_process)} files to process.")
        for filename in files_to_process:
            path = os.path.join(data_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line)
                        # เพิ่มข้อมูลแหล่งที่มาเข้าไปใน item เพื่อใช้อ้างอิง
                        item['_source_filename'] = filename
                        item['_source_line_num'] = line_num
                        if item.get("content"):
                            all_items.append(item)
                    except json.JSONDecodeError: 
                        print(f"  - ⚠️ Skipping malformed JSON on line {line_num} in '{filename}'")
                        continue
        
        print(f"📦 Total documents loaded: {len(all_items)}")
        return all_items

    def build_and_save_index(self, all_items: List[Dict], index_folder: str):
        print(f"\n--- 🏭 Building a unified index ---")
        os.makedirs(index_folder, exist_ok=True)
        
        texts_to_embed = []
        mapping_data = []
        
        for item in all_items:
            book = item.get("book_title", "N/A")
            content = item.get("content", "")
            context_str = f"จากหนังสือ '{book}'"
            embedding_text = f"query: {context_str}, เนื้อหา: {content}"
            texts_to_embed.append(embedding_text)
            mapping_data.append(item)

        if not texts_to_embed:
            print(f"  - 🟡 No text to index. Stopping.")
            return

        print(f"  - 🧠 Generating {len(texts_to_embed)} embeddings (using {str(self.model.device).upper()})...")
        embeddings = self.model.encode(
            texts_to_embed, 
            convert_to_numpy=True, 
            show_progress_bar=True
        ).astype("float32")
        
        # สร้าง Index เดียว
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        faiss.write_index(index, os.path.join(index_folder, "faiss_index.bin"))
        
        mapping_filepath = os.path.join(index_folder, "faiss_mapping.json")
        with open(mapping_filepath, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=4)
                
        print(f"  - ✅ Unified index saved successfully to '{index_folder}'.")

if __name__ == "__main__":
    SOURCE_DATA_FOLDER = "data/source"
    INDEX_FOLDER = "data/index"

    print("\n" + "="*60)
    print("--- 🛠️  Starting RAG Knowledge Base Construction  🛠️ ---")
    print("="*60)

    builder = RAGIndexBuilder()
    all_book_data = builder.load_all_data(data_folder=SOURCE_DATA_FOLDER)
    
    if all_book_data:
        builder.build_and_save_index(all_book_data, index_folder=INDEX_FOLDER)
    
    print("\n" + "="*60)
    print("✅ RAG build process finished successfully!")
    print("="*60)