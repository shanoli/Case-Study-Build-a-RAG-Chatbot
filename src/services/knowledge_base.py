"""
Knowledge Base Service - Task 1
Handles document loading, chunking, embedding, and ChromaDB operations
"""
import hashlib
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import KnowledgeBaseException, EmbeddingException

logger = get_logger(__name__)


class KnowledgeBaseService:
    """
    Service for managing knowledge base with vector embeddings
    
    Features:
    - Text chunking with semantic awareness
    - Vector embeddings using text-embedding-004
    - ChromaDB persistence
    - Duplicate detection
    - Multi-lingual support
    """
    
    def __init__(self):
        """Initialize knowledge base service"""
        # Configure Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        self.collection_name = "knowledge_base"
        
        # Text splitter with semantic boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )
        
        logger.info(
            "knowledge_base_initialized",
            embedding_model=settings.EMBEDDING_MODEL,
            chunk_size=settings.CHUNK_SIZE
        )
    
    async def initialize(self):
        """Initialize or load existing collection"""
        try:
            # Get or create collection with cosine similarity
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",  # Cosine similarity for vector search
                    "hnsw:construction_ef": 200,  # Higher quality index
                    "hnsw:M": 16  # Number of connections per layer
                }
            )
            
            count = self.collection.count()
            logger.info(
                "collection_loaded",
                collection=self.collection_name,
                chunks=count
            )
            
        except Exception as e:
            logger.error("collection_init_failed", error=str(e))
            raise KnowledgeBaseException(f"Failed to initialize collection: {e}")
    
    def _generate_chunk_id(self, text: str, index: int) -> str:
        """
        Generate unique ID for chunk based on content hash
        
        Args:
            text: Chunk text
            index: Chunk index
        
        Returns:
            Unique chunk ID
        """
        hash_obj = hashlib.md5(text.encode('utf-8'))
        return f"chunk_{index}_{hash_obj.hexdigest()[:12]}"
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using text-embedding-004
        
        This uses Google's latest embedding model with:
        - 768 dimensions
        - Multi-lingual support
        - Improved semantic understanding
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors (one per input text)
        """
        try:
            all_embeddings = []
            
            # Batch processing for efficiency
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings using Gemini API
                # genai.embed_content() is a synchronous function, call it directly
                result = genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"  # Optimized for document retrieval
                )
                
                # result['embedding'] returns a list of vectors, one per input text
                batch_embeddings = result['embedding']
                
                # Ensure all_embeddings is a list of vectors
                for embedding_vector in batch_embeddings:
                    all_embeddings.append(embedding_vector)
                
                logger.debug(
                    "embeddings_generated",
                    batch_num=i // batch_size + 1,
                    batch_size=len(batch),
                    embedding_dims=len(batch_embeddings[0]) if batch_embeddings else 0
                )
            
            logger.info(
                "all_embeddings_complete",
                total_embeddings=len(all_embeddings),
                dimensions=len(all_embeddings[0]) if all_embeddings else 0
            )
            
            return all_embeddings
            
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e), exception_type=type(e).__name__)
            raise EmbeddingException(f"Failed to generate embeddings: {e}")
    
    def _parse_product_info(self, content: str) -> List[Dict[str, Any]]:
        """Parse product info file into semantic units with metadata"""
        chunks = []
        current_section = "intro"  # Start with intro for the summary
        
        lines = content.split('\n')
        
        # Section mapping (more flexible matching)
        section_headers = {
            "catalog and inventory summary": "summary",
            "detailed product list": "product",
            "return policies": "policy",
            "customer support": "support"
        }
        
        i = 0
        summary_lines = []
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
                
            # Check for section header
            is_header = False
            lower_line = line.lower()
            for header, section_type in section_headers.items():
                if header in lower_line and (line.startswith('#') or line.startswith('##')):
                    
                    # If we were in intro/summary, save it now
                    if current_section == "summary" and summary_lines:
                        chunks.append({
                            "content": "\n".join(summary_lines),
                            "metadata": {"type": "summary", "product_name": "all"}
                        })
                        summary_lines = []
                        
                    current_section = section_type
                    is_header = True
                    break
            
            if is_header or line.startswith('---'):
                i += 1
                continue
                
            # Handle sections
            if current_section == "summary" or current_section == "intro":
                if current_section == "intro": current_section = "summary"
                summary_lines.append(line)
            
            elif current_section == "product":
                if re.match(r'^\d+\.\s+Product:', line):
                    product_block = [line]
                    i += 1
                    while i < len(lines):
                        next_line = lines[i].strip()
                        if re.match(r'^\d+\.\s+Product:', next_line) or next_line.startswith('#'):
                            break
                        if next_line:
                            product_block.append(next_line)
                        i += 1
                    
                    full_text = "\n".join(product_block)
                    name_match = re.search(r'Product:\s*(.*?)(?:\n|$)', full_text)
                    chunks.append({
                        "content": full_text,
                        "metadata": {"type": "product", "product_name": name_match.group(1).strip() if name_match else "unknown"}
                    })
                    continue
                    
            elif current_section in ["policy", "support"]:
                if re.match(r'^\d+\.\s+', line) or line.startswith('*'):
                    chunks.append({
                        "content": line,
                        "metadata": {"type": current_section}
                    })
            
            i += 1
            
        # Add final summary if exists
        if current_section == "summary" and summary_lines:
            chunks.append({
                "content": "\n".join(summary_lines),
                "metadata": {"type": "summary", "product_name": "all"}
            })
            
        return chunks

    async def upload_knowledge_base(
        self,
        content: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        replace_existing: bool = False,
        source: str = "upload"
    ) -> Dict[str, Any]:
        """
        Upload and process knowledge base with vector embeddings
        
        Args:
            content: Text content to process
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            replace_existing: Clear existing data before upload
            source: Source identifier for metadata
        
        Returns:
            Upload statistics
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "upload_started",
                content_length=len(content),
                replace_existing=replace_existing
            )
            
            # Clear existing if requested
            if replace_existing:
                await self.clear()
            
            # Prepare chunks and metadatas
            chunks = []
            metadatas = []
            
            # Use semantic splitting for product info
            if source == "product_info.txt" or "## Products" in content:
                parsed_items = self._parse_product_info(content)
                logger.info("semantic_parsing_used", items_found=len(parsed_items))
                
                for i, item in enumerate(parsed_items):
                    chunks.append(item["content"])
                    meta = item["metadata"]
                    meta.update({
                        "chunk_index": i,
                        "created_at": datetime.now().isoformat(),
                        "source": source,
                        "chunk_size": len(item["content"]),
                        "language": "multi"
                    })
                    metadatas.append(meta)
            else:
                # Configure text splitter
                if chunk_size:
                    self.text_splitter.chunk_size = chunk_size
                if chunk_overlap:
                    self.text_splitter.chunk_overlap = chunk_overlap
                
                # Split text into semantically meaningful chunks
                chunks = self.text_splitter.split_text(content)
                logger.info("text_chunked", num_chunks=len(chunks))
                
                metadatas = [
                    {
                        "chunk_index": i,
                        "created_at": datetime.now().isoformat(),
                        "source": source,
                        "chunk_size": len(chunk),
                        "language": "multi"
                    }
                    for i, chunk in enumerate(chunks)
                ]
            
            # Generate vector embeddings using text-embedding-004
            embeddings = await self._embed_texts(chunks)
            logger.info(
                "embeddings_created",
                num_embeddings=len(embeddings),
                dimensions=len(embeddings[0]) if embeddings else 0
            )
            
            # Prepare data for ChromaDB
            ids = [self._generate_chunk_id(chunk, i) for i, chunk in enumerate(chunks)]
            
            # Check for duplicates (idempotent uploads)
            existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
            
            new_ids = []
            new_chunks = []
            new_embeddings = []
            new_metadatas = []
            duplicates_skipped = 0
            
            for id_, chunk, embedding, metadata in zip(ids, chunks, embeddings, metadatas):
                if id_ not in existing_ids:
                    new_ids.append(id_)
                    new_chunks.append(chunk)
                    new_embeddings.append(embedding)
                    new_metadatas.append(metadata)
                else:
                    duplicates_skipped += 1
            
            # Add to ChromaDB for vector search
            if new_ids:
                self.collection.add(
                    ids=new_ids,
                    documents=new_chunks,
                    embeddings=new_embeddings,
                    metadatas=new_metadatas
                )
                
                logger.info(
                    "chunks_added_to_vector_db",
                    new_chunks=len(new_ids),
                    total_chunks=self.collection.count()
                )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "status": "success",
                "chunks_created": len(new_ids),
                "embeddings_generated": len(new_embeddings),
                "duplicates_skipped": duplicates_skipped,
                "processing_time_ms": int(processing_time),
                "model_used": settings.EMBEDDING_MODEL,
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0
            }
            
            logger.info("upload_completed", **result)
            return result
            
        except Exception as e:
            logger.error("upload_failed", error=str(e))
            raise KnowledgeBaseException(f"Upload failed: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics
        
        Returns:
            Status information
        """
        try:
            count = self.collection.count()
            
            # Get sample embedding to check dimensions
            sample_dim = 0
            if count > 0:
                sample = self.collection.get(limit=1, include=["embeddings"])
                if sample["embeddings"]:
                    sample_dim = len(sample["embeddings"][0])
            
            # Estimate storage size (approximate)
            storage_mb = count * 0.05  # ~50KB per chunk with embeddings
            
            return {
                "total_chunks": count,
                "total_documents": 1,
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimensions": sample_dim,
                "last_updated": datetime.now(),
                "storage_size_mb": round(storage_mb, 2),
                "index_ready": count > 0,
                "vector_search_enabled": True,
                "similarity_metric": "cosine"
            }
            
        except Exception as e:
            logger.error("status_check_failed", error=str(e))
            raise KnowledgeBaseException(f"Status check failed: {e}")
    
    async def clear(self):
        """Clear all knowledge base data"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                }
            )
            logger.info("knowledge_base_cleared")
            
        except Exception as e:
            logger.error("clear_failed", error=str(e))
            raise KnowledgeBaseException(f"Clear failed: {e}")
    
    async def get_embedding_details(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get embedding details for specific chunk
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Embedding details or None
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not result["ids"]:
                return None
            
            return {
                "chunk_id": chunk_id,
                "text": result["documents"][0],
                "embedding_dimension": len(result["embeddings"][0]),
                "embedding_preview": result["embeddings"][0][:10],
                "metadata": result["metadatas"][0]
            }
            
        except Exception as e:
            logger.error("get_embedding_failed", chunk_id=chunk_id, error=str(e))
            raise KnowledgeBaseException(f"Get embedding failed: {e}")
