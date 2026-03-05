# core/rag_engine.py
# Optimized RAG Engine: loads embedder once at startup, uses BAAI-bge-m3, no reranker.
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any


class BookRAGEngine:
    """
    RAG Engine using FAISS for retrieval with BAAI-bge-m3 embeddings.
    
    Key design decisions:
    - Embedder is loaded ONCE in __init__ and reused across all requests.
    - Runs on CPU to preserve GPU VRAM for the LLM (~7.4GB / 8GB).
    - CrossEncoder reranker removed to reduce latency and memory usage;
      bge-m3 produces high-quality embeddings that make reranking less necessary.
    """

    def __init__(self, index_path: str = "data/index", 
                 embedder_path: str = "/home/mikedev/MyModels/Model-RAG/BAAI-bge-m3"):
        print("⚙️  Book RAG Engine is initializing...")
        self.device = "cpu"  # Keep on CPU — GPU VRAM reserved for LLM

        # Load embedder ONCE at startup (previously loaded per-request, ~5-10s wasted each time)
        print(f"  - 🧠 Loading Embedder '{embedder_path}' on {self.device.upper()}...")
        self.embedder = SentenceTransformer(embedder_path, device=self.device)
        print("  - ✅ Embedder loaded successfully.")

        # Load FAISS index and mapping
        self.index, self.mapping = None, None
        self._load_unified_index(index_path)

        print("✅ Book RAG Engine is ready.")

    def _load_unified_index(self, path: str):
        """Load the pre-built FAISS index and its document mapping from disk."""
        print(f"  - 📚 Loading Knowledge Base from '{path}'...")
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

            print(f"    - ✅ Knowledge Base with {len(self.mapping)} documents loaded.")

        except Exception as e:
            print(f"    - ❌ CRITICAL: Error loading book index: {e}")

    def _deduplicate_passages(self, results: List[tuple], min_diff_len: int = 50) -> List[tuple]:
        """Remove near-duplicate passages based on book title + content prefix."""
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

    def search(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant passages.
        
        Pipeline (simplified from previous version):
        1. Encode query with bge-m3 embedder (cached, no reload)
        2. FAISS similarity search (top_k * 3 candidates for dedup headroom)
        3. Deduplicate passages
        4. Return top_k unique results
        
        Removed: CrossEncoder reranking step (saves ~3-5s per request)
        """
        if not self.index or not self.mapping:
            return {"context": "Error: Knowledge base is not loaded.", "sources": [], "best_score": 0.0}

        # Step 1: Encode query using the CACHED embedder (no disk load)
        query_vector = self.embedder.encode(
            ["query: " + query], 
            convert_to_numpy=True
        )

        # Step 2: FAISS search — retrieve extra candidates for deduplication headroom
        retrieval_count = top_k * 3
        distances, indices = self.index.search(query_vector, retrieval_count)

        if distances.size > 0:
            best_distance = distances[0][0]
            best_score = 1 / (1 + best_distance)
        else:
            best_score = 0.0

        # Step 3: Map indices back to document data
        retrieved_candidates = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(self.mapping):
                item = self.mapping[idx]
                distance = float(distances[0][rank])
                similarity = 1 / (1 + distance)
                retrieved_candidates.append((similarity, item))

        if not retrieved_candidates:
            return {"context": "ไม่พบข้อมูลที่เกี่ยวข้องโดยตรงจากคลังความรู้", "sources": [], "best_score": best_score}

        # Step 4: Deduplicate and take top_k
        unique_results = self._deduplicate_passages(retrieved_candidates, min_diff_len=50)
        top_results = unique_results[:top_k]

        print(f"  - [RAG] Returning {len(top_results)} unique results (best_score: {best_score:.4f}).")

        # Step 5: Format context string for the LLM
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