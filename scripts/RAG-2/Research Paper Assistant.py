# -*- coding: utf-8 -*-
"""Executable Python version of RAG-2/Research Paper Assistant.ipynb."""

# Install dependencies from requirements-rag.txt before running.


import os
import glob
import re
import json
import hashlib
import pandas as pd
import fitz

from langchain_core.documents import Document
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Set USE_DOCLING=1 to use the optional Docling parser instead of PyMuPDF.
USE_DOCLING = os.getenv("USE_DOCLING", "0") == "1"
if USE_DOCLING:
    from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Works from either the project root or RAG-2 as the notebook working directory.
PDF_FOLDER = next((str(p.resolve()) for p in [Path("RAG-2/ResearchPapers"), Path("ResearchPapers")] if p.is_dir()), None)
if PDF_FOLDER is None:
    raise FileNotFoundError("ResearchPapers folder not found; see README.md")

pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
print("Total PDFs found:", len(pdf_files))

if not pdf_files:
    raise FileNotFoundError("No PDF files found. Check the PDF_FOLDER path.")

all_documents = []
paper_info = {}

for pdf_path in pdf_files:
    pdf = fitz.open(pdf_path)
    pdf_metadata = pdf.metadata or {}
    total_pages = len(pdf)
    filename = os.path.basename(pdf_path)

    paper_title = pdf_metadata.get("title")
    if not paper_title or paper_title.strip() == "":
        paper_title = os.path.splitext(filename)[0]

    paper_id = hashlib.md5(os.path.abspath(pdf_path).encode()).hexdigest()[:8]

    paper_info[paper_id] = {
        "title": paper_title,
        "filename": filename,
        "path": os.path.abspath(pdf_path),
        "pages": total_pages
    }
    pdf.close()

    if USE_DOCLING:
        loader = DoclingLoader(file_path=pdf_path)
        loaded_docs = loader.load()
    else:
        # Lightweight PDF extraction, without downloading large Docling models.
        with fitz.open(pdf_path) as document:
            loaded_docs = [
                Document(page_content=page.get_text("text"), metadata={"page_number": n + 1})
                for n, page in enumerate(document) if page.get_text("text").strip()
            ]

    for doc in loaded_docs:
        doc.metadata["paper_id"] = paper_id
        doc.metadata["paper_title"] = paper_title
        doc.metadata["filename"] = filename
        doc.metadata["pdf_path"] = os.path.abspath(pdf_path)
        all_documents.append(doc)

print("Loaded document sections:", len(all_documents))

def get_page_number(metadata):
    possible_keys = ["page", "page_number", "page_no", "page_index"]

    for key in possible_keys:
        value = metadata.get(key)
        if value is not None:
            try:
                page_number = int(value)
                if key == "page_index":
                    page_number += 1
                return page_number
            except (ValueError, TypeError):
                pass

    provenance = metadata.get("dl_meta", {})
    if isinstance(provenance, dict):
        for key in possible_keys:
            value = provenance.get(key)
            if value is not None:
                try:
                    page_number = int(value)
                    if key == "page_index":
                        page_number += 1
                    return page_number
                except (ValueError, TypeError):
                    pass
    return None


for doc in all_documents:
    page_number = get_page_number(doc.metadata)
    doc.metadata["page_number"] = page_number if page_number is not None else "Unknown"

print("Ingestion Summary")
print("Number of papers:", len(paper_info))

total_pages = sum(info["pages"] for info in paper_info.values())
print("Total pages:", total_pages)

for paper_id, info in paper_info.items():
    paper_documents = [
        doc for doc in all_documents
        if doc.metadata["paper_id"] == paper_id
    ]
    print("\nPaper ID:", paper_id)
    print("Paper Title:", info["title"])
    print("Pages:", info["pages"])
    print("Docling documents:", len(paper_documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(all_documents)
print("Total chunks:", len(chunks))

for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = str(index)
    if chunk.metadata.get("page_number") is None:
        chunk.metadata["page_number"] = "Unknown"

chunk_summary = []

for paper_id, info in paper_info.items():
    paper_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["paper_id"] == paper_id
    ]
    chunk_summary.append({
        "Paper ID": paper_id,
        "Paper Title": info["title"],
        "Pages": info["pages"],
        "Chunks": len(paper_chunks)
    })

chunk_summary_df = pd.DataFrame(chunk_summary)
print(chunk_summary_df.to_string(index=False))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_PATH = str(Path(PDF_FOLDER).parent / "rag_db")
COLLECTION_NAME = "research_papers"

def create_chunk_id(chunk):
    text = chunk.page_content
    paper_id = chunk.metadata["paper_id"]
    page_no = chunk.metadata.get("page_number", "Unknown")
    unique_text = f"{paper_id}_{page_no}_{text}"
    return hashlib.md5(unique_text.encode("utf-8")).hexdigest()

for chunk in chunks:
    chunk.metadata["chunk_id"] = create_chunk_id(chunk)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

if chunks:
    vector_store.add_documents(
        documents=chunks,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks]
    )

print("Chunks stored in ChromaDB:", len(chunks))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def single_paper_retrieval(query, paper_id, k=4):
    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"paper_id": paper_id}
    )
    return results

def cross_paper_retrieval(query, k_per_paper=2):
    all_results = []
    for paper_id in paper_info.keys():
        paper_results = vector_store.similarity_search(
            query,
            k=k_per_paper,
            filter={"paper_id": paper_id}
        )
        all_results.extend(paper_results)
    return all_results

from getpass import getpass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    GROQ_API_KEY = getpass("Enter your Groq API key: ")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research paper assistant.
Answer the user's question using only retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not invent paper titles, page numbers, results, or citations.
3. If the answer is not in the retrieved context, say that the context does not contain the answer.

Retrieved context:
{context}
"""
    ),
    ("human", "Question: {question}")
])

rag_chain = prompt | llm | StrOutputParser()

def build_context(documents):
    context = ""
    for index, doc in enumerate(documents):
        paper_title = doc.metadata.get("paper_title", "Unknown")
        page_no = doc.metadata.get("page_number", "Unknown")
        context += f"""
Chunk {index + 1}
Paper: {paper_title}
Page: {page_no}
Content:
{doc.page_content}
"""
    return context

def generate_answer(query, retrieved_docs):
    context = build_context(retrieved_docs)
    answer = rag_chain.invoke({
        "context": context,
        "question": query
    })
    return answer

def print_sources(retrieved_docs):
    print("Sources:")
    seen_sources = set()

    for doc in retrieved_docs:
        paper_title = doc.metadata.get("paper_title", "Unknown")
        paper_no = doc.metadata.get("page_number", "Unknown")
        source = (paper_title, paper_no)

        if source not in seen_sources:
            seen_sources.add(source)
            print(f"{paper_title} | Page {paper_no}")

def ans_question(query, mode="cross", paper_id=None):
    if mode == "single":
        if paper_id is None:
            raise ValueError("Single paper ID is required.")
        retrieved_docs = single_paper_retrieval(query, paper_id)
    else:
        retrieved_docs = cross_paper_retrieval(query)

    answer = generate_answer(query, retrieved_docs)
    print("\nAnswer:")
    print(answer)
    print_sources(retrieved_docs)
    return answer

while True:
    print("\n===== Research Paper Assistant =====")
    mode = input("Choose mode (single/cross/exit): ").strip().lower()

    if mode == "exit":
        print("Exiting Research Assistant.")
        break

    if mode not in ["single", "cross"]:
        print("Invalid mode.")
        continue

    query = input("Enter your research question: ").strip()
    if query.lower() == "exit":
        break
    if not query:
        print("Please enter a question.")
        continue

    paper_id = None
    if mode == "single":
        for pid, info in paper_info.items():
            print(f"{pid}: {info['title']}")
        paper_id = input("Enter Paper ID: ").strip()
        if paper_id not in paper_info:
            print("Invalid Paper ID.")
            continue

    ans_question(query, mode=mode, paper_id=paper_id)
