"""
Custom exceptions for the RAG chatbot
"""


class RAGChatbotException(Exception):
    """Base exception for RAG Chatbot"""
    pass


class KnowledgeBaseException(RAGChatbotException):
    """Exception related to knowledge base operations"""
    pass


class EmbeddingException(RAGChatbotException):
    """Exception related to embedding generation"""
    pass


class RetrievalException(RAGChatbotException):
    """Exception related to retrieval"""
    pass


class GenerationException(RAGChatbotException):
    """Exception related to LLM generation"""
    pass


class SessionException(RAGChatbotException):
    """Exception related to session management"""
    pass


class ClassificationException(RAGChatbotException):
    """Exception related to intent classification"""
    pass
