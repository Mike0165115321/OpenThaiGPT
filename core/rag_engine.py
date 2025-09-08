# core/rag_engine.py (Corrected and Improved Version)
import faiss
import json
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from typing import List, Dict, Any

class BookRAGEngine:
    def __init__(self, index_path: str = "data/index"):
        print("⚙️  Book RAG Engine (CPU-Mode) is initializing...")
        self.device = "cpu"
        self.embedder_model_name = "intfloat/multilingual-e5-large"
        self.reranker_model_name = 'BAAI/bge-reranker-base'

        self.index, self.mapping = None, None
        self._load_unified_index(index_path)

        print("✅ Book RAG Engine is ready.")

    def _load_unified_index(self, path: str):
        print(f"  - 📚 Loading Unified Book Knowledge Base from '{path}'...")
        if not os.path.exists(path): 
            print("    - ❌ CRITICAL: Index path not found. RAG will not function.")
            return
        try:
            index_filepath = os.path.join(path, "faiss_index.bin")
            mapping_filepath = os.path.join(path, "faiss_mapping.json")

            if not os.path.exists(index_filepath) or not os.path.exists(mapping_filepath):
                print(f"    - ❌ CRITICAL: Index or mapping file missing in '{path}'.")
                return

            self.index = faiss.read_index(index_filepath)
            with open(mapping_filepath, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
            
            print(f"    - ✅ Unified Knowledge Base with {len(self.mapping)} documents is ready.")

        except Exception as e:
            print(f"    - ❌ CRITICAL: Error loading book index: {e}")
    
    def _deduplicate_passages(self, results, min_diff_len: int = 30):
        seen = set()
        unique_results = []
        for score, item in results:
            content = item.get("content", "").strip()
            if not content:
                continue
            key = (item.get("book_title"), content[:min_diff_len])
            if key not in seen:
                seen.add(key)
                unique_results.append((score, item))
        return unique_results

    def search(self, query: str, top_k_retrieval: int = 20, top_k_rerank: int = 6) -> Dict[str, Any]:
        if not self.index or not self.mapping:
            return {"context": "Error: Knowledge base is not loaded.", "sources": [], "best_score": 0.0}
        
        embedder = SentenceTransformer(self.embedder_model_name, device=self.device)
        query_vector = embedder.encode(["query: " + query], convert_to_numpy=True)
        del embedder
        
        distances, indices = self.index.search(query_vector, top_k_retrieval)
        
        if distances.size > 0:
            best_distance = distances[0][0]
            best_score = 1 / (1 + best_distance)
        else:
            best_score = 0.0

        retrieved_candidates = [self.mapping[i] for i in indices[0] if i < len(self.mapping)]
        
        if not retrieved_candidates:
            return {"context": "ไม่พบข้อมูลที่เกี่ยวข้องโดยตรงจากคลังความรู้", "sources": [], "best_score": best_score}
        
        reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        sentence_pairs = [[query, item.get('content', '')] for item in retrieved_candidates]
        scores = reranker.predict(sentence_pairs, show_progress_bar=False)
        del reranker

        reranked_results = sorted(zip(scores, retrieved_candidates), key=lambda x: x[0], reverse=True)
        
        unique_reranked_results = self._deduplicate_passages(reranked_results, min_diff_len=50)
        top_results = unique_reranked_results[:top_k_rerank]

        print(f"  - [RAG] Step 4: Formatting final context from {len(top_results)} unique results.")

        final_contexts = []
        final_sources = set()
        book_seen = set()

        for score, item in top_results:
            book_title = item.get("book_title", "Unknown Source")
            if book_title in book_seen:
                continue  
            content = item.get("content", "")
            context_str = f"จากหนังสือ '{book_title}' กล่าวว่า:\n\"\"\"\n{content}\n\"\"\""
            final_contexts.append(context_str)
            final_sources.add(book_title)
            book_seen.add(book_title)
            
        final_context_str = "\n\n---\n\n".join(final_contexts)
        
        return {
            "context": final_context_str, 
            "sources": sorted(list(final_sources)),
            "best_score": float(best_score) 
        }