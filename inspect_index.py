# inspect_index.py
import faiss
import numpy as np
import os

# --- Configuration ---
# ชี้ไปที่ไฟล์ index ของคุณ
index_filepath = "data/index/faiss_index.bin"

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Inspecting FAISS Index: {index_filepath} ---")

    if not os.path.exists(index_filepath):
        print(f"❌ Error: Index file not found at '{index_filepath}'")
    else:
        try:
            # 1. โหลด index จากไฟล์
            print("Step 1: Loading index from file...")
            index = faiss.read_index(index_filepath)
            print("✅ Index loaded successfully.")

            # 2. แสดงข้อมูลพื้นฐานของ Index
            print("\n--- Index Information ---")
            print(f"Number of vectors in the index (index.ntotal): {index.ntotal}")
            print(f"Vector dimension (index.d): {index.d}")
            print(f"Is the index trained? (index.is_trained): {index.is_trained}")

            # 3. ดึงเวกเตอร์ตัวอย่างออกมาดู (สำคัญ)
            # เราสามารถ "reconstruct" หรือสร้างเวกเตอร์ดั้งเดิมกลับคืนมาจาก index ได้
            if index.ntotal > 0:
                print("\n--- Reconstructing Vectors ---")
                
                # ดึงเวกเตอร์ตัวแรก (ID ที่ 0)
                vector_id_to_check = 0
                print(f"Reconstructing vector for ID: {vector_id_to_check}...")
                
                reconstructed_vector = index.reconstruct(vector_id_to_check)
                
                print(f"✅ Vector for ID {vector_id_to_check} (first 10 dimensions):")
                print(reconstructed_vector[:10]) # แสดงแค่ 10 มิติแรกพอ ไม่ให้ล้นจอ
                print(f"Shape of the reconstructed vector: {reconstructed_vector.shape}")

            else:
                print("Index is empty, no vectors to reconstruct.")

        except Exception as e:
            print(f"❌ An error occurred while reading the index: {e}")