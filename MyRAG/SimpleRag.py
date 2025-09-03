"""
Minimal RAG app using:
- LangChain (LCEL), FAISS vector store, OpenAI chat + embeddings
- A Retriever and a "chain over documents" (stuffing) combine step
- Optional LangSmith tracing (set env vars to enable)

Usage:
    python rag_app.py "What does the document say about X?"
If no CLI question is given, it will prompt you interactively.

First run: builds FAISS index from ./docs
Later runs: loads the cached FAISS index in ./faiss_index/
"""

import os
import sys
from pathlib import Path
from typing import List

# --- LangChain core pieces (v0.2+ style) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- OpenAI chat + embeddings via langchain-openai ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Vector store + loaders from community ---
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# -----------------------------
# Configuration / Paths
# -----------------------------
DOCS_DIR = Path("./docs")
INDEX_DIR = Path("./faiss_index")  # FAISS saves index + metadata here
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# Endpoint for LangSmith API'1.1.1-GettingStarted.ipynb
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# -----------------------------
# Helpers: load documents
# -----------------------------
def load_documents_from_folder(folder: Path) -> List[Document]:
    """
    Load .txt, .md, .pdf from a folder into LangChain Documents.
    Extend with more loaders as needed.
    """
    docs: List[Document] = []

    if not folder.exists():
        print(f"[info] {folder} does not exist; creating it. Add some files and re-run.")
        folder.mkdir(parents=True, exist_ok=True)
        return docs

    for path in folder.glob("**/*"):
        if path.is_dir():
            continue

        ext = path.suffix.lower()
        try:
            if ext in [".txt", ".md"]:
                # Each file -> one Document
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            elif ext == ".pdf":
                # Each page -> one Document
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())
            # Add more types (docx/csv/html) with appropriate loaders if you need
        except Exception as e:
            print(f"[warn] Skipping {path.name}: {e}")

    return docs


def build_or_load_vectorstore(embedding) -> FAISS:
    """
    Build a FAISS index from docs/ on first run; otherwise load it from disk.
    """
    if INDEX_DIR.exists():
        print("[info] Loading existing FAISS index from disk...")
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=embedding,
            allow_dangerous_deserialization=True,  # needed for local load
        )

    print("[info] Building FAISS index from docs...")
    raw_docs = load_documents_from_folder(DOCS_DIR)
    if not raw_docs:
        # If no docs found, create a tiny in-memory sample so the app still works.
        print("[info] No docs found in ./docs. Using a tiny built-in sample.")
        raw_docs = [
            Document(
                page_content=(
                    "This is a sample knowledge base about RAG. "
                    "RAG stands for Retrieval-Augmented Generation. "
                    "It retrieves relevant chunks from a vector store and 'augments' the prompt to a language model."
                ),
                metadata={"source": "sample"},
            )
        ]

    # Chunking for better retrieval recall
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # tune per domain
        chunk_overlap=120,   # small overlap helps continuity
    )
    chunks = splitter.split_documents(raw_docs)

    # Build FAISS vector store
    vs = FAISS.from_documents(chunks, embedding)

    # Persist to disk so we don't rebuild each run
    vs.save_local(str(INDEX_DIR))
    print(f"[info] Saved FAISS index to {INDEX_DIR.resolve()}")
    return vs


def build_rag_chain():
    """
    Create a retrieval pipeline:
      retriever -> stuff-docs chain -> ChatOpenAI
    This uses LCEL 'create_stuff_documents_chain' + 'create_retrieval_chain'.
    """
    # 1) LLM + Embeddings
    llm = ChatOpenAI(  # gpt-4o-mini is a great default; adjust if you want
        model="gpt-4o-mini",
        temperature=0.1,
    )
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2) Vector store / retriever
    vectorstore = build_or_load_vectorstore(embedding)
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # MMR = diverse results
        search_kwargs={"k": 4},
    )

    # 3) Prompt for “stuff” (chain over docs)
    #    We’ll show sources back to the user as well.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know. Be concise.",
            ),
            (
                "human",
                "Question:\n{input}\n\n"
                "Context (top retrieved chunks):\n{context}\n\n"
                "Answer:"
            ),
        ]
    )

    # 4) Build a chain that “stuffs” the retrieved docs into the prompt
    #    (other options include map_reduce or refine for long contexts)
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # 5) Make a full retrieval chain: (inputs) -> retriever -> doc_chain
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain, retriever


def run_query(question: str):
    """
    Execute the RAG pipeline with the given question.
    Prints the answer and the source documents used.
    """
    rag_chain, retriever = build_rag_chain()

    # Call the chain. The retrieval chain expects dict input with "input"
    result = rag_chain.invoke({"input": question})

    # The standard return has 'answer' and 'context' (the docs)
    answer = result.get("answer") or result.get("output_text")  # depending on versions
    context_docs: List[Document] = result.get("context", [])

    print("\n=== Answer ===")
    print(answer.strip() if answer else "(no answer)")

    if context_docs:
        print("\n=== Sources ===")
        for i, d in enumerate(context_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            page_str = f" (page {page})" if page is not None else ""
            snippet = d.page_content[:200].replace("\n", " ")
            print(f"[{i}] {src}{page_str}: {snippet}...")
    else:
        print("\n(no sources returned)")


def main():
    # Support both CLI and interactive use
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your question (Ctrl+C to exit):")
        question = input("> ").strip()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the script."
        )

    # Optional: LangSmith tracing is enabled purely by env vars.
    # If LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, runs will be visible in your project.
    # Nothing else is needed here.

    run_query(question)


if __name__ == "__main__":
    main()
