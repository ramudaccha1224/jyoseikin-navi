import os
import json
from PyPDF2 import PdfReader

def create_chunks_from_pdf(pdf_path, chunk_size=500, overlap=100):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text.replace('\n', ' ')
    
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append({"source": os.path.basename(pdf_path), "content": chunk})
        start += (chunk_size - overlap)
    return chunks

# 3. すべてのPDFを対象に処理を実行
pdf_files = [f for f in os.listdir(".") if f.lower().endswith('.pdf')]
all_chunks = []

for pdf_file in pdf_files:
    print(f"Breaking down {pdf_file} for RAG...")
    all_chunks.extend(create_chunks_from_pdf(pdf_file))

# 4. 保存するファイル名：pdf_chunks.json（上書きされます）
with open("pdf_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Success: 合計 {len(all_chunks)} 個の断片を保存しました。")