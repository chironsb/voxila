"""
Retrieval-Augmented Generation (RAG) module
Uses sentence-transformers for embeddings and FAISS for vector search
"""
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional


class RAG:
    """RAG system with embeddings and vector search"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-small-en-v1.5", 
                 vault_file: str = "./data/vault.txt",
                 faiss_index_path: str = "./data/embeddings/faiss_index"):
        """
        Initialize the RAG system
        
        Args:
            embedding_model: Sentence transformer model name
            vault_file: Path to the knowledge base file
            faiss_index_path: Path to save/load FAISS index
        """
        self.embedding_model_name = embedding_model
        self.vault_file = vault_file
        self.faiss_index_path = faiss_index_path
        self.embedding_model = None
        self.index = None
        self.vault_content = []
        self.dimension = None
        
        self._load_embedding_model()
        self._load_or_create_index()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        # Get embedding dimension
        test_embedding = self.embedding_model.encode(["test"])
        self.dimension = test_embedding.shape[1]
        print(f"Embedding model loaded. Dimension: {self.dimension}")
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        # Ensure vault file directory exists
        vault_dir = os.path.dirname(self.vault_file)
        if vault_dir and not os.path.exists(vault_dir):
            os.makedirs(vault_dir, exist_ok=True)
        
        # Create vault file if it doesn't exist
        if not os.path.exists(self.vault_file):
            with open(self.vault_file, 'w', encoding='utf-8') as f:
                pass  # Create empty file
        
        index_file = f"{self.faiss_index_path}.index"
        content_file = f"{self.faiss_index_path}.content.pkl"
        
        # Ensure FAISS index directory exists
        index_dir = os.path.dirname(self.faiss_index_path)
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)
        
        if os.path.exists(index_file) and os.path.exists(content_file):
            print(f"Loading existing FAISS index from {index_file}...")
            self.index = faiss.read_index(index_file)
            with open(content_file, 'rb') as f:
                self.vault_content = pickle.load(f)
            print(f"Loaded {len(self.vault_content)} documents from index.")
        else:
            print("Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.vault_content = []
            # Index existing vault if it exists
            if os.path.exists(self.vault_file):
                self.index_vault()
    
    def _save_index(self):
        """Save the FAISS index and content to disk"""
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        index_file = f"{self.faiss_index_path}.index"
        content_file = f"{self.faiss_index_path}.content.pkl"
        
        faiss.write_index(self.index, index_file)
        with open(content_file, 'wb') as f:
            pickle.dump(self.vault_content, f)
        print(f"Saved FAISS index to {index_file}")
    
    def index_vault(self):
        """Index the vault.txt file"""
        if not os.path.exists(self.vault_file):
            print(f"Vault file not found: {self.vault_file}")
            return
        
        print(f"Indexing vault file: {self.vault_file}")
        
        # Read vault content
        with open(self.vault_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Filter out empty lines and comments
        content_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                content_lines.append(line)
        
        if not content_lines:
            print("No content to index in vault file.")
            return
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(content_lines, show_progress_bar=True)
        
        # Clear existing index and content
        self.index.reset()
        self.vault_content = []
        
        # Add to FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index.add(embeddings_np)
        self.vault_content = content_lines
        
        # Save index
        self._save_index()
        print(f"Indexed {len(content_lines)} documents.")
    
    def add_document(self, text: str):
        """
        Add a single document to the index
        
        Args:
            text: Document text to add
        """
        if not text.strip():
            return
        
        # Generate embedding
        embedding = self.embedding_model.encode([text])
        embedding_np = np.array(embedding).astype('float32')
        
        # Add to index
        self.index.add(embedding_np)
        self.vault_content.append(text)
        
        # Save index
        self._save_index()
        print(f"Added document to index. Total documents: {len(self.vault_content)}")
    
    def remove_all_documents(self):
        """Remove all documents from the index"""
        self.index.reset()
        self.vault_content = []
        self._save_index()
        print("Removed all documents from index.")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        if len(self.vault_content) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        # Search in FAISS
        top_k = min(top_k, len(self.vault_content))
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        # Return results with similarity scores
        # FAISS returns L2 distances, convert to similarity (lower distance = higher similarity)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.vault_content):
                # Convert distance to similarity (1 / (1 + distance))
                similarity = 1 / (1 + dist)
                results.append((self.vault_content[idx], similarity))
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get relevant context for a query
        
        Args:
            query: Search query
            top_k: Number of top results to include
            
        Returns:
            Combined context string
        """
        results = self.search(query, top_k)
        if not results:
            return ""
        
        context_parts = [f"- {doc}" for doc, score in results]
        return "\n".join(context_parts)

