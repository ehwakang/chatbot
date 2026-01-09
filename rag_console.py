import os
from pathlib import Path
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.text_splitter import CharacterTextSplitter


PDF_DIR = Path("./pdfs")
EMBED_MODEL_NAME = "jhgan/ko-sroberta-multitask" # 변경 가능


# 1) PDF -> 텍스트
def extract_text_from_pdf(path: Path) -> str:
  text = ""
  with fitz.open(path) as doc:
    for page in doc:
      text += page.get_text()
  return text


# 2) 로드
def load_documents(pdf_dir: Path) -> List[Dict]:
  docs = []
  for p in pdf_dir.glob("*.pdf"):
    t = extract_text_from_pdf(p)
    docs.append({"text": t, "meta": {"source": str(p.name)}})
    return docs


# 3) 청킹
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)




def build_vectorstore(documents: List[Dict]):
  texts = []
  metadatas = []
  for d in documents:
    parts = splitter.split_text(d["text"])
    for i, p in enumerate(parts):
      texts.append(p)
      metadatas.append({**d["meta"], "chunk": i})


  # 임베딩
  emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
  db = LangchainFAISS.from_texts(texts, emb, metadatas=metadatas)
  return db




if __name__ == "__main__":
  print("Loading PDFs...")
  docs = load_documents(PDF_DIR)
  if not docs:
    print("No PDFs found in ./pdfs. Put sample PDFs there and rerun.")
    exit(1)


  print(f"Loaded {len(docs)} documents. Building vectorstore...")
  db = build_vectorstore(docs)


  # LLM: OpenAI (환경변수 OPENAI_API_KEY 필요)
  llm = OpenAI(temperature=0)
  qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k":3}))


  print("Ready. Ask questions (type 'exit' to quit):")
  while True:
    q = input("Q> ")
    if q.strip().lower() in ("exit","quit"):
      break
    ans = qa.run(q)
    print("A>", ans)