"""
Vector Store using ChromaDB

Manages document storage and retrieval using vector embeddings.
ChromaDB is an embedded database - no separate server needed for simplicity and making sire that I'm still the instuction constrants.

Key Operations:
1. Add documents (text -> embedding -> storage)
2. Search (query -> embedding -> find similar -> return docs)
3. Persist  dor reuse
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from src.retrieval.embeddings import get_embedding_generator
from src.utils.token_counter import get_token_counter


class VectorStore:
    """
    Manages document storage and retrieval using ChromaDB.
    
    ChromaDB stores documents as vector embeddings, enabling
    semantic search (finding by meaning, not just keywords).
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize vector store.
        """
        load_dotenv()
        
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.persist_directory = persist_directory or os.getenv(
            'CHROMA_PERSIST_DIRECTORY',
            './data/vectorstore'
        )
        self.collection_name = collection_name or os.getenv(
            'CHROMA_COLLECTION_NAME',
            'rag_corpus'
        )
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG corpus for context-aware system"}
        )
        
        # Initialize embedding generator
        self.embedding_generator = get_embedding_generator()
        self.token_counter = get_token_counter()
        
        self.logger.info(
            f"Initialized VectorStore: {self.collection.count()} documents in collection"
        )
    
    def add_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a single document to the vector store.
        """
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_embedding(content)
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata['token_count'] = self.token_counter.count_tokens(content)
            
            # Add to collection
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[doc_metadata]
            )
            
            self.logger.info(f"Added document: {document_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add document {document_id}: {str(e)}")
            raise
    
    def add_documents_batch(
        self,
        documents: List[Tuple[str, str, Optional[Dict]]]
    ) -> None:
        """
        Add multiple documents efficiently.
        
        Args:
            documents: List of (id, content, metadata) tuples
        """
        if not documents:
            return
        
        ids = []
        contents = []
        metadatas = []
        
        for doc_id, content, metadata in documents:
            ids.append(doc_id)
            contents.append(content)
            
            doc_metadata = metadata or {}
            doc_metadata['token_count'] = self.token_counter.count_tokens(content)
            metadatas.append(doc_metadata)
        
        # Generate embeddings in batch
        embeddings = self.embedding_generator.generate_embeddings_batch(contents)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
        
        self.logger.info(f"Added {len(documents)} documents in batch")
    
    def search(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for documents similar to query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of result dictionaries with keys:
            - id: Document ID
            - content: Document text
            - score: Similarity score (0-1, higher is more similar)
            - metadata: Document metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    #Chromadb returns a distance(lower = more similar)
                    #We need to convert to similariyt score(higher = more similar)
                    #for L2 distance,similaruty - 1/(1+distance)
                    #for cosine distance,similarity = 1-distance
                    distance = results['distances'][0][i]
                    #distance-->similaryt score
                    similarity = 1.0 / (1.0 + distance)

                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'score': similarity,  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {}
                    })
            
            self.logger.info(f"Search returned {len(formatted_results)} results")
           # return formatted_results
            if formatted_results:
                scores = [r['score'] for r in formatted_results]
                self.logger.debug(f"Similarity scores:{scores}")
            return formatted_results    
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
    
    def delete_document(self, document_id: str) -> None:
        """
        Delete a document from the store.
        
        Args:
            document_id: ID of document to delete
        """
        try:
            self.collection.delete(ids=[document_id])
            self.logger.info(f"Deleted document: {document_id}")
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        Warning: This deletes all data!
        """
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "RAG corpus for context-aware system"}
        )
        self.logger.warning("Cleared all documents from collection")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            'document_count': count,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory
        }
    
    def load_corpus_from_directory(self, corpus_dir: str) -> int:
        """
        Load all text files from a directory into the vector store.
        
        Args:
            corpus_dir: Path to directory containing .txt files
            
        Returns:
            Number of documents loaded
        """
        corpus_path = Path(corpus_dir)
        
        if not corpus_path.exists():
            self.logger.error(f"Corpus directory not found: {corpus_dir}")
            return 0
        
        # Find all .txt files
        txt_files = list(corpus_path.glob("*.txt"))
        
        if not txt_files:
            self.logger.warning(f"No .txt files found in {corpus_dir}")
            return 0
        
        self.logger.info(f"Loading {len(txt_files)} documents from {corpus_dir}")
        
        # Prepare documents
        documents = []
        for txt_file in txt_files:
            try:
                content = txt_file.read_text(encoding='utf-8').strip()
                if not content:
                    self.logger.warning(f"Skipping empty file: {txt_file.name}")
                doc_id = txt_file.stem  # Filename without extension
                metadata = {
                    'source': str(txt_file),
                    'filename': txt_file.name
                }
                documents.append((doc_id, content, metadata))
                
            except Exception as e:
                self.logger.error(f"Failed to read {txt_file}: {str(e)}")
        
        # Add in batch
        if documents:
            self.add_documents_batch(documents)
        
        return len(documents)


# Global singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get the global VectorStore instance.
    
    Returns:
        Singleton VectorStore instance
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance

