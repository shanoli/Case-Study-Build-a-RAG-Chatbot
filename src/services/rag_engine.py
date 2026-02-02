"""
RAG Engine - Task 2
Retrieval-Augmented Generation with hybrid search (semantic + BM25 keywords)
"""
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import google.generativeai as genai
from rank_bm25 import BM25Okapi

from src.services.knowledge_base import KnowledgeBaseService
from src.core.config import settings
from src.core.logging import get_logger
from src.core.exceptions import RetrievalException, GenerationException
from src.utils.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    FALLBACK_RESPONSE,
    detect_language
)

logger = get_logger(__name__)


class RAGEngine:
    """
    RAG chain with HYBRID SEARCH (semantic + keyword-based)
    
    Features:
    - Vector similarity search using cosine distance (semantic)
    - BM25 keyword-based search (sparse vectors)
    - Hybrid score fusion (weighted combination)
    - Context relevance filtering
    - Multi-lingual response generation
    - Fallback handling
    - Retry logic for robustness
    
    Hybrid Search Benefits:
    - Semantic matching for conceptual similarity
    - Keyword matching for exact term matches
    - Better recall and precision
    - Works with low-threshold vectors
    """
    
    def __init__(self):
        """Initialize RAG engine with hybrid search"""
        # Configure Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.kb_service = KnowledgeBaseService()
        self.relevance_threshold = settings.RELEVANCE_THRESHOLD
        
        # Hybrid search weights
        self.semantic_weight = 0.6  # 60% semantic vector similarity
        self.keyword_weight = 0.4   # 40% BM25 keyword relevance
        
        # BM25 index (built on first retrieval)
        self.bm25_index = None
        self.all_chunks = None
        self.all_metadatas = None
        
        # Initialize Gemini model for generation
        self.generation_config = {
            "temperature": settings.TEMPERATURE,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
        }
        
        self.model = genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            generation_config=self.generation_config,
        )
        
        logger.info(
            "rag_engine_initialized",
            model=settings.GEMINI_MODEL,
            threshold=self.relevance_threshold,
            search_method="hybrid_semantic_bm25",
            semantic_weight=self.semantic_weight,
            keyword_weight=self.keyword_weight
        )
    
    async def initialize(self):
        """Initialize knowledge base and build BM25 index"""
        try:
            await self.kb_service.initialize()
            
            # Build BM25 index from all chunks for keyword search
            await self._build_bm25_index()
            
            logger.info("rag_engine_kb_initialized", bm25_ready=self.bm25_index is not None)
        except Exception as e:
            logger.error("rag_engine_kb_initialization_failed", error=str(e))
            raise
    
    async def refresh_index(self):
        """Rebuild the BM25 index from current KB contents"""
        logger.info("refreshing_rag_engine_index")
        await self._build_bm25_index()

    async def _build_bm25_index(self):
        """
        Build BM25 index from knowledge base
        
        Called once on initialization to prepare keyword search
        """
        try:
            if not self.kb_service.collection:
                return
            
            # Get all documents from collection
            all_docs = self.kb_service.collection.get(include=["documents", "metadatas"])
            
            if not all_docs or not all_docs.get("documents"):
                logger.warning("no_documents_for_bm25_index")
                return
            
            self.all_chunks = all_docs["documents"]
            self.all_metadatas = all_docs["metadatas"]
            
            # Tokenize: simple lowercase split + remove stopwords
            corpus_tokens = [
                self._tokenize(doc) for doc in self.all_chunks
            ]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(corpus_tokens)
            
            logger.info(
                "bm25_index_built",
                num_documents=len(self.all_chunks),
                avg_doc_len=sum(len(tokens) for tokens in corpus_tokens) / len(corpus_tokens)
            )
            
        except Exception as e:
            logger.error("bm25_index_build_failed", error=str(e))
            self.bm25_index = None
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization with stopword removal
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens
        """
        # Simple stopwords in English (can be expanded)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'is', 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she'
        }
        
        # Tokenize: lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation and filter stopwords
        filtered = []
        for token in tokens:
            # Remove punctuation
            clean = ''.join(c for c in token if c.isalnum())
            
            # Keep if not stopword and not empty
            if clean and clean not in stopwords and len(clean) > 1:
                filtered.append(clean)
        
        return filtered
    
    def _get_bm25_scores(self, query: str, top_k: int = 10) -> Dict[int, float]:
        """
        Get BM25 relevance scores for query
        
        Args:
            query: Search query
            top_k: Number of top results
        
        Returns:
            Dictionary of {chunk_index: bm25_score}
        """
        if not self.bm25_index or not self.all_chunks:
            logger.debug("bm25_index_not_ready")
            return {}
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                logger.debug("query_has_no_meaningful_tokens")
                return {}
            
            # Get BM25 scores for all documents
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            result = {idx: float(scores[idx]) for idx in top_indices if scores[idx] > 0}
            
            logger.debug(
                "bm25_scores_computed",
                query=query,
                num_results=len(result),
                top_score=max(result.values()) if result else 0.0
            )
            
            return result
            
        except Exception as e:
            logger.warning("bm25_scoring_failed", error=str(e))
            return {}
    
    async def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for search query
        
        Args:
            query: Search query
        
        Returns:
            Query embedding vector
        """
        try:
            result = genai.embed_content(
                model=settings.EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"  # Optimized for query
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error("query_embedding_failed", error=str(e))
            raise RetrievalException(f"Query embedding failed: {e}")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        search_method: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks using HYBRID SEARCH
        
        Combines two search strategies:
        1. SEMANTIC SEARCH: Vector similarity using text-embedding-004
           - Captures conceptual meaning
           - Good for paraphrased queries
           - Weight: 60%
        
        2. KEYWORD SEARCH: BM25 algorithm
           - Captures exact term matches
           - Good for specific product names
           - Weight: 40%
        
        Hybrid Fusion:
        - Combines both scores with weights
        - Normalizes scores for fair comparison
        - Returns top-k by fused score
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score (overrides default)
            search_method: "hybrid" (default), "semantic_only", or "keyword_only"
        
        Returns:
            Retrieved chunks with hybrid scores and breakdown
        """
        start_time = datetime.now()
        threshold = min_score or self.relevance_threshold
        
        try:
            logger.info(
                "retrieval_started",
                query=query,
                top_k=top_k,
                search_method=search_method,
                filters=filters
            )
            
            # Ensure knowledge base is initialized
            if not self.kb_service.collection:
                await self.initialize()
            
            # ===== SEMANTIC SEARCH =====
            semantic_results = {}
            try:
                query_embedding = await self._embed_query(query)
                
                # Apply filters if provided
                query_kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": top_k * 2,
                    "include": ["documents", "metadatas", "distances"]
                }
                
                if filters:
                    query_kwargs["where"] = filters
                
                results = self.kb_service.collection.query(**query_kwargs)
                
                if results.get("documents") and results["documents"][0]:
                    logger.debug("raw_chroma_results", count=len(results["documents"][0]), ids=results["ids"][0])
                    for idx, (doc, metadata, distance) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0]
                    )):
                        # Convert distance to similarity
                        semantic_score = 1 - distance
                        
                        # Store with chunk_index or idx if missing
                        doc_key = metadata.get("chunk_index", idx)
                        # Ensure key is unique by using ID if index is duplicated
                        if doc_key in semantic_results:
                            doc_key = results["ids"][0][idx]
                            
                        semantic_results[doc_key] = {
                            "content": doc,
                            "metadata": metadata,
                            "semantic_score": semantic_score,
                            "id": results["ids"][0][idx]
                        }
                
                logger.debug(
                    "semantic_search_completed",
                    results_found=len(semantic_results)
                )
                
            except Exception as e:
                logger.warning("semantic_search_failed", error=str(e))
                if search_method == "semantic_only":
                    raise
            
            # ===== KEYWORD SEARCH (BM25) =====
            keyword_results = {}
            try:
                bm25_scores = self._get_bm25_scores(query, top_k * 5)
                
                # Merge keyword scores into results
                for chunk_idx, bm25_score in bm25_scores.items():
                    if chunk_idx < len(self.all_chunks):
                        # Apply filters to BM25 results
                        if filters and self.all_metadatas:
                            metadata = self.all_metadatas[chunk_idx]
                            matches = True
                            for f_key, f_val in filters.items():
                                meta_val = metadata.get(f_key)
                                # Check for $in operator
                                if isinstance(f_val, dict) and "$in" in f_val:
                                    if meta_val not in f_val["$in"]:
                                        matches = False
                                        break
                                elif meta_val != f_val:
                                    matches = False
                                    break
                            
                            if not matches:
                                continue
                        
                        chunk_content = self.all_chunks[chunk_idx]
                        # Map to Chromadb ID for cross-reference
                        chroma_id = self.all_ids[chunk_idx] if hasattr(self, 'all_ids') else str(chunk_idx)
                        
                        keyword_results[chroma_id] = {
                            "content": chunk_content,
                            "metadata": self.all_metadatas[chunk_idx] if self.all_metadatas else {"chunk_index": chunk_idx},
                            "keyword_score": bm25_score,
                            "id": chroma_id
                        }
                
                logger.debug(
                    "keyword_search_completed",
                    results_found=len(keyword_results)
                )
                
            except Exception as e:
                logger.warning("keyword_search_failed", error=str(e))
                if search_method == "keyword_only":
                    raise
            
            # ===== HYBRID FUSION =====
            fused_results = {}
            
            if search_method == "hybrid":
                # Combine semantic and keyword results
                all_keys = set(semantic_results.keys()) | set(keyword_results.keys())
                
                for key in all_keys:
                    semantic_score = 0.0
                    keyword_score = 0.0
                    
                    if key in semantic_results:
                        semantic_score = semantic_results[key]["semantic_score"]
                    if key in keyword_results:
                        # Normalize BM25 score to 0-1 range (using a more dynamic denominator)
                        # BM25 scores can vary, 20 is a safer denominator for many documents
                        keyword_score = min(keyword_results[key]["keyword_score"] / 20.0, 1.0)
                    
                    # Weighted fusion: 60% semantic + 40% keyword
                    fused_score = (
                        semantic_score * self.semantic_weight +
                        keyword_score * self.keyword_weight
                    )
                    
                    # Get content from whichever source has it
                    content = (semantic_results.get(key) or keyword_results.get(key))["content"]
                    metadata = (semantic_results.get(key) or keyword_results.get(key))["metadata"]
                    
                    fused_results[key] = {
                        "content": content,
                        "metadata": metadata,
                        "fused_score": fused_score,
                        "semantic_score": semantic_score,
                        "keyword_score": keyword_score
                    }
                
                logger.debug(
                    "hybrid_fusion_completed",
                    total_candidates=len(fused_results)
                )
                
            elif search_method == "semantic_only":
                fused_results = {
                    k: {**v, "fused_score": v["semantic_score"]}
                    for k, v in semantic_results.items()
                }
            else:  # keyword_only
                fused_results = {
                    k: {
                        **v,
                        "fused_score": min(v["keyword_score"] / 20.0, 1.0),
                        "semantic_score": 0.0
                    }
                    for k, v in keyword_results.items()
                }
            
            # Sort by fused score
            sorted_results = sorted(
                fused_results.items(),
                key=lambda x: x[1]["fused_score"],
                reverse=True
            )
            
            # Filter by relevance threshold (more lenient for hybrid)
            hybrid_threshold = threshold * 0.75  # 25% more lenient for hybrid
            
            chunks = []
            
            # DEBUG: Print top 5 candidates and their scores
            logger.info("retrieval_candidates_debug", 
                       query=query,
                       threshold=hybrid_threshold,
                       top_5=[
                {"content": r[1]["content"][:50], "fused": r[1]["fused_score"], "semantic": r[1].get("semantic_score"), "keyword": r[1].get("keyword_score")}
                for r in sorted_results[:5]
            ])
            
            for key, result in sorted_results[:top_k]:
                score = result["fused_score"]
                
                if score >= hybrid_threshold:
                    chunks.append({
                        "content": result["content"],
                        "fused_score": round(score, 4),
                        "semantic_score": round(result.get("semantic_score", 0), 4),
                        "keyword_score": round(result.get("keyword_score", 0), 4),
                        "metadata": result["metadata"]
                    })
                    
                    logger.debug(
                        "chunk_selected",
                        fused_score=score,
                        semantic_score=result.get("semantic_score", 0),
                        keyword_score=result.get("keyword_score", 0),
                        chunk_preview=result["content"][:80]
                    )
            
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            avg_score = sum(c["fused_score"] for c in chunks) / len(chunks) if chunks else 0.0
            
            response = {
                "query": query,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "avg_score": round(avg_score, 4),
                "retrieval_time_ms": int(retrieval_time),
                "search_method": search_method,
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
                "fusion_threshold": hybrid_threshold if search_method == "hybrid" else threshold
            }
            
            if not chunks:
                response["message"] = (
                    f"No relevant context found "
                    f"(best scores below {response['fusion_threshold']} threshold)"
                )
                logger.info(
                    "no_relevant_chunks",
                    query=query,
                    threshold=response["fusion_threshold"]
                )
            else:
                logger.info(
                    "retrieval_completed",
                    chunks_found=len(chunks),
                    avg_score=avg_score,
                    search_method=search_method
                )
            
            return response
            
        except Exception as e:
            logger.error("retrieval_failed", query=query, error=str(e))
            raise RetrievalException(f"Retrieval failed: {e}")
    
    async def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for search query
        
        Args:
            query: Search query
        
        Returns:
            Query embedding vector
        """
        try:
            result = genai.embed_content(
                model=settings.EMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"  # Optimized for query
            )
            
            return result['embedding']
            
        except Exception as e:
            logger.error("query_embedding_failed", error=str(e))
            raise RetrievalException(f"Query embedding failed: {e}")
    
    async def generate(
        self,
        query: str,
        context: str = "",
        conversation_history: Optional[List[Dict]] = None,
        use_rag: bool = True,
        detected_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate response using Gemini with multi-lingual support
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous messages
            use_rag: Whether to use RAG context
            detected_language: Detected language code
        
        Returns:
            Generated response with metadata
        """
        start_time = datetime.now()
        
        try:
            logger.info(
                "generation_started",
                query=query,
                has_context=bool(context),
                language=detected_language
            )
            
            # Build conversation history
            chat_history = []
            
            # Add system instruction
            chat_history.append({
                "role": "user",
                "parts": [RAG_SYSTEM_PROMPT]
            })
            chat_history.append({
                "role": "model",
                "parts": ["I understand. I will assist customers with product information, pricing, warranty, returns, and support in their preferred language."]
            })
            
            # Add previous conversation (last 5 exchanges)
            if conversation_history:
                for msg in conversation_history[-10:]:  # Last 10 messages (5 exchanges)
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history.append({
                        "role": role,
                        "parts": [msg["content"]]
                    })
            
            # Build current query with context
            if use_rag and context:
                current_message = RAG_USER_PROMPT.format(
                    context=context,
                    question=query
                )
            else:
                current_message = query
            
            # Generate response with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create chat session
                    chat = self.model.start_chat(history=chat_history)
                    
                    # Generate response
                    response = await asyncio.to_thread(
                        chat.send_message,
                        current_message
                    )
                    
                    generated_text = response.text
                    
                    # Extract usage metadata
                    tokens_used = 0
                    if hasattr(response, 'usage_metadata'):
                        tokens_used = response.usage_metadata.total_token_count
                    
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    
                    logger.info(
                        "generation_completed",
                        tokens=tokens_used,
                        latency_ms=int(latency),
                        language=detected_language
                    )
                    
                    return {
                        "generated_text": generated_text,
                        "model": settings.GEMINI_MODEL,
                        "tokens_used": tokens_used,
                        "latency_ms": int(latency),
                        "context_used": bool(context),
                        "fallback": False,
                        "language": detected_language
                    }
                    
                except Exception as e:
                    logger.warning(
                        "generation_attempt_failed",
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    
                    if attempt == max_retries - 1:
                        raise
                    
                    # Exponential backoff
                    await asyncio.sleep(1 * (attempt + 1))
            
        except Exception as e:
            logger.error("generation_failed", query=query, error=str(e))
            
            # Fallback response in detected language
            fallback_text = FALLBACK_RESPONSE.get(
                detected_language,
                FALLBACK_RESPONSE["en"]
            )
            
            return {
                "generated_text": fallback_text,
                "model": settings.GEMINI_MODEL,
                "tokens_used": 0,
                "latency_ms": 0,
                "context_used": False,
                "fallback": True,
                "error": str(e),
                "language": detected_language
            }
    
    async def rag_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate
        
        This is the main RAG pipeline:
        1. Detect language
        2. Embed query
        3. Vector search for relevant chunks
        4. Generate contextual response
        
        Args:
            query: User query
            conversation_history: Conversation history
            top_k: Number of chunks to retrieve
        
        Returns:
            Generated answer with metadata
        """
        try:
            # Detect language for multi-lingual support
            detected_language = detect_language(query)
            logger.info("language_detected", language=detected_language)
            
            # Step 1: Retrieve context using VECTOR SEARCH
            retrieval_result = await self.retrieve(query, top_k=top_k, filters=filters)
            
            # Step 2: Build context from retrieved chunks
            context = "\n\n".join([
                f"[Source {i+1}] {chunk['content']}"
                for i, chunk in enumerate(retrieval_result["chunks"])
            ])
            
            # Step 3: Generate answer
            if context:
                generation_result = await self.generate(
                    query=query,
                    context=context,
                    conversation_history=conversation_history,
                    use_rag=True,
                    detected_language=detected_language
                )
            else:
                # No relevant context - use fallback
                fallback_text = FALLBACK_RESPONSE.get(
                    detected_language,
                    FALLBACK_RESPONSE["en"]
                )
                
                generation_result = {
                    "generated_text": fallback_text,
                    "model": settings.GEMINI_MODEL,
                    "tokens_used": 0,
                    "latency_ms": 0,
                    "context_used": False,
                    "fallback": True,
                    "language": detected_language
                }
            
            # Combine results
            return {
                **generation_result,
                "retrieved_chunks": retrieval_result["total_chunks"],
                "avg_relevance": retrieval_result["avg_score"],
                "vector_search_used": True
            }
            
        except Exception as e:
            logger.error("rag_query_failed", query=query, error=str(e))
            raise GenerationException(f"RAG query failed: {e}")
