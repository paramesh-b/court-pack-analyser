"""
RAG Pipeline for Court Pack Analyser
-------------------------------------
Chunks document text → embeds with sentence-transformers (all-MiniLM-L6-v2)
→ stores persistently in ChromaDB → retrieves semantically relevant context.

Filenames are stored in doc_names.csv inside chroma_db so the Document
Library shows real filenames instead of hash IDs.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb
import hashlib
import os

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR      = "./chroma_db"
METADATA_FILE   = os.path.join(CHROMA_DIR, "doc_names.csv")

CLAIM_QUERIES = {
    "hire_rate":    "daily hire rate charged cost per day GBP",
    "vehicle":      "vehicle category type car model class",
    "duration":     "hire duration days rental period length",
    "claimant":     "claimant name policyholder plaintiff",
    "region":       "region location area national London",
    "hire_company": "hire company credit hire firm supplier name",
    "total_amount": "total claim amount invoice cost GBP",
}


def _doc_id(text: str) -> str:
    return "doc_" + hashlib.md5(text.encode()).hexdigest()[:12]


def _load_names() -> dict:
    if not os.path.exists(METADATA_FILE):
        return {}
    rows = {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "," in line:
                    k, v = line.split(",", 1)
                    rows[k.strip()] = v.strip()
    except Exception:
        pass
    return rows


def _save_name(collection_name: str, filename: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    rows = _load_names()
    rows[collection_name] = filename
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            for k, v in rows.items():
                f.write(f"{k},{v}\n")
    except Exception:
        pass


class CourtPackRAG:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.vectorstore     = None
        self.collection_name = None
        os.makedirs(CHROMA_DIR, exist_ok=True)

    def index_document(self, text: str, filename: str = "Unknown") -> int:
        """
        Index document into ChromaDB.
        Always saves the filename — even if collection already exists.
        Returns number of chunks.
        """
        self.collection_name = _doc_id(text)

        # Always save/update the filename mapping first
        _save_name(self.collection_name, filename)

        client   = chromadb.PersistentClient(path=CHROMA_DIR)
        existing = [c.name for c in client.list_collections()]

        if self.collection_name in existing:
            # Reuse existing collection — no re-embedding needed
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=CHROMA_DIR,
            )
            return self.vectorstore._collection.count()

        # New document — chunk, embed, persist
        chunks = self.splitter.split_text(text)
        if not chunks:
            raise ValueError("Document produced no chunks.")

        documents = [
            Document(
                page_content=chunk,
                metadata={"chunk_id": i, "filename": filename}
            )
            for i, chunk in enumerate(chunks)
        ]

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=CHROMA_DIR,
        )
        return len(chunks)

    def load_existing(self, collection_name: str) -> int:
        """Load a previously indexed document by collection name."""
        self.collection_name = collection_name
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )
        return self.vectorstore._collection.count()

    def retrieve(self, query: str, k: int = 3) -> str:
        if self.vectorstore is None:
            raise RuntimeError("No document indexed.")
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_claim_context(self, k: int = 2) -> dict:
        return {
            field: self.retrieve(query, k=k)
            for field, query in CLAIM_QUERIES.items()
        }

    @staticmethod
    def list_indexed_documents() -> list[dict]:
        """Return all documents with real filenames for the sidebar."""
        if not os.path.exists(CHROMA_DIR):
            return []
        try:
            client    = chromadb.PersistentClient(path=CHROMA_DIR)
            names_map = _load_names()
            return [
                {
                    "collection_name": c.name,
                    "filename":        names_map.get(c.name, c.name),
                    "chunk_count":     c.count(),
                }
                for c in client.list_collections()
            ]
        except Exception:
            return []

    @staticmethod
    def delete_document(collection_name: str) -> bool:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            client.delete_collection(collection_name)
            rows = _load_names()
            rows.pop(collection_name, None)
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                for k, v in rows.items():
                    f.write(f"{k},{v}\n")
            return True
        except Exception:
            return False