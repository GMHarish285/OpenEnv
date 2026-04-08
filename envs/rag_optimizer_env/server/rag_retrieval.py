"""
RAG Retrieval Module using ChromaDB and HotpotQA Knowledge Base.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb


class RAGRetriever:
    """Retrieves documents from the HotpotQA knowledge base."""
    
    def __init__(self, kb_path: str = "./knowledge_base"):
        """
        Initialize the RAG retriever.
        
        Args:
            kb_path: Path to the ChromaDB persistence directory
        """
        self.kb_path = kb_path
        self.embedder = None
        self.collection = None
        self._initialized = False
        
    def initialize(self):
        """Lazy initialization of the retriever (call after KB is built)."""
        if self._initialized:
            return
            
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(
                f"Knowledge base not found at {self.kb_path}. "
                f"Run 'python server/build_kb.py' first to build the KB."
            )
        
        print(f"Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Connecting to ChromaDB at {self.kb_path}...")
        client = chromadb.PersistentClient(path=self.kb_path)
        self.collection = client.get_collection(name="hotpotqa_kb")
        
        print(f" RAG Retriever initialized with {self.collection.count()} documents")
        self._initialized = True
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k most relevant documents for a query.
        
        Args:
            query: The question or search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of dicts with 'title', 'text', and 'score' keys
        """
        if not self._initialized:
            self.initialize()
        
        # Embed the query
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "title": meta["title"],
                "text": doc,
                "score": 1 - dist  # convert distance to similarity
            })
        
        return retrieved
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into a context string for LLM prompting.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"Document {i}: {doc['title']}\n{doc['text']}"
            )
        return "\n\n".join(context_parts)


# Singleton instance
_retriever = None

def get_retriever() -> RAGRetriever:
    """Get or create the global RAG retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
