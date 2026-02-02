"""
LangGraph Workflow - Task 3
Stateful conversation orchestration with intent classification and routing
"""
from typing import Dict, List, Any, Optional, Literal
from typing_extensions import TypedDict
from datetime import datetime
import re
import traceback

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from src.services.rag_engine import RAGEngine
from src.core.config import settings
from src.core.logging import get_logger
from src.utils.prompts import (
    CLASSIFIER_PROMPT,
    GREETING_RESPONSES,
    ESCALATION_RESPONSES,
    detect_language
)

logger = get_logger(__name__)


# State Schema for LangGraph
class ConversationState(TypedDict):
    """
    State schema for conversation workflow
    
    This state is passed between nodes in the LangGraph
    """
    # Input
    message: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    
    # Classification
    category: str
    confidence: float
    entities: Dict[str, Any]
    language: str
    
    # RAG
    retrieved_chunks: List[Dict[str, Any]]
    context_used: bool
    
    # Output
    reply: str
    source: str
    metadata: Dict[str, Any]


class IntentClassifier:
    """
    Intent classification node
    
    Classifies user queries into categories:
    - products: Product information queries
    - returns: Return/refund policy queries
    - general: General FAQ queries
    - greeting: Greetings and small talk
    - unknown: Unclear or out-of-scope queries
    """
    
    CATEGORY_KEYWORDS = {
        "products": [
            "product", "item", "buy", "purchase", "price", "cost",
            "available", "stock", "sell", "offer", "catalog", "feature",
            "specification", "model", "brand", "warranty", "guarantee",
            "watch", "smartwatch", "earbud", "power bank", "pro x", "lite s", "elite", "ultra"
        ],
        "returns": [
            "return", "refund", "exchange", "cancel", "money back",
            "send back", "replacement", "defective", "damaged",
            "policy", "wrong item", "damaged item", "broken"
        ],
        "general": [
            "contact", "support", "help", "hours", "location",
            "shipping", "delivery", "payment", "order", "track",
            "customer service", "email", "phone", "hours", "time"
        ],
        "greeting": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "how are you", "what's up",
            "namaste", "namaskar", "morning", "evening"
        ]
    }
    
    def __init__(self):
        """Initialize classifier"""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.0,  # Low temperature for consistent classification
            google_api_key=settings.GOOGLE_API_KEY
        )
        logger.info("intent_classifier_initialized")
    
    def _keyword_classify(self, message: str) -> tuple[str, float]:
        """
        Keyword-based classification (fast fallback)
        
        Args:
            message: User message
        
        Returns:
            (category, confidence_score)
        """
        message_lower = message.lower()
        
        category_scores = {}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in message_lower)
            category_scores[category] = matches
        
        # Find best match
        if max(category_scores.values()) > 0:
            best_category = max(category_scores, key=category_scores.get)
            # Higher confidence if multiple keywords match
            confidence = min(0.5 + (category_scores[best_category] * 0.1), 0.95)
            return best_category, confidence
        
        return "unknown", 0.0

    async def _llm_classify(self, message: str) -> tuple[str, float]:
        """
        LLM-based classification for ambiguous queries
        """
        try:
            prompt = CLASSIFIER_PROMPT.format(message=message)
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip().lower()
            
            # Simple parsing of LLM output (assuming it returns just the category)
            valid_categories = ["products", "returns", "general", "greeting", "unknown"]
            
            for cat in valid_categories:
                if cat in content:
                    return cat, 0.85
                    
            return "unknown", 0.3
        except Exception as e:
            logger.error("llm_classification_failed", error=str(e))
            return "unknown", 0.0
    
    def _extract_entities(self, message: str, category: str) -> Dict[str, Any]:
        """
        Extract entities from message
        
        Args:
            message: User message
            category: Detected category
        
        Returns:
            Extracted entities
        """
        entities = {}
        
        # Extract product mentions
        if category == "products":
            # Simple product name extraction
            product_pattern = r'\b(?:laptop|phone|tablet|watch|headphone|speaker|camera)\w*\b'
            products = re.findall(product_pattern, message, re.IGNORECASE)
            if products:
                entities["products"] = list(set(products))
        
        # Extract price mentions
        price_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d{2})?'
        prices = re.findall(price_pattern, message)
        if prices:
            entities["prices"] = prices
        
        return entities
    
    async def classify(self, state: ConversationState) -> ConversationState:
        """
        Classify user intent
        
        This is Node 1 in the LangGraph workflow
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with classification
        """
        message = state["message"]
        
        logger.info("classification_started", message=message[:100])
        
        try:
            # Detect language
            language = detect_language(message)
            
            # Try keyword classification first
            category, confidence = self._keyword_classify(message)
            
            # If keyword classification is uncertain, use LLM
            if category == "unknown" or confidence < 0.6:
                logger.info("using_llm_classification", prev_cat=category, prev_conf=confidence)
                llm_category, llm_confidence = await self._llm_classify(message)
                
                # Only override if LLM is more confident
                if llm_confidence > confidence:
                    category = llm_category
                    confidence = llm_confidence
            
            # Extract entities
            entities = self._extract_entities(message, category)
            
            # Update state
            state["category"] = category
            state["confidence"] = confidence
            state["entities"] = entities
            state["language"] = language
            
            logger.info(
                "classification_completed",
                category=category,
                confidence=confidence,
                language=language
            )
            
            return state
            
        except Exception as e:
            logger.error("classification_failed", error=str(e), traceback=traceback.format_exc())
            
            # Fallback to unknown
            state["category"] = "unknown"
            state["confidence"] = 0.0
            state["entities"] = {}
            state["language"] = "en"
            
            return state


class RAGResponderNode:
    """
    RAG-based response generation node
    
    Retrieves context from vector store and generates response
    """
    
    def __init__(self):
        """Initialize RAG responder"""
        self.rag_engine = RAGEngine()
        logger.info("rag_responder_initialized")
    
    async def respond(self, state: ConversationState) -> ConversationState:
        """
        Generate RAG-based response
        
        This is Node 2 in the LangGraph workflow
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with RAG response
        """
        message = state["message"]
        conversation_history = state.get("conversation_history", [])
        
        logger.info("rag_response_started", message=message[:100])
        
        try:
            # Map category to metadata filters
            category = state.get("category")
            
            # Apply metadata filters based on intent
            filters = None
            if category == "products":
                # For product queries, include both specific products and the summary
                filters = {"type": {"$in": ["product", "summary"]}}
            elif category == "warranty":
                filters = {"type": "policy"}
            elif category == "support":
                filters = {"type": "support"}
                
            # Execute RAG query
            result = await self.rag_engine.rag_query(
                query=message,
                conversation_history=conversation_history,
                top_k=5,
                filters=filters
            )
            
            # Update state
            state["reply"] = result["generated_text"]
            state["source"] = "rag"
            state["context_used"] = result["context_used"]
            state["metadata"] = {
                "retrieved_chunks": result["retrieved_chunks"],
                "avg_relevance": result["avg_relevance"],
                "tokens_used": result["tokens_used"],
                "latency_ms": result["latency_ms"],
                "model": result["model"],
                "fallback": result.get("fallback", False)
            }
            
            logger.info(
                "rag_response_completed",
                chunks_used=result["retrieved_chunks"],
                context_used=result["context_used"]
            )
            
            return state
            
        except Exception as e:
            logger.error("rag_response_failed", error=str(e), traceback=traceback.format_exc())
            
            # Fallback response
            language = state.get("language", "en")
            from src.utils.prompts import FALLBACK_RESPONSE
            
            state["reply"] = FALLBACK_RESPONSE.get(language, FALLBACK_RESPONSE["en"])
            state["source"] = "fallback"
            state["context_used"] = False
            state["metadata"] = {"error": str(e)}
            
            return state


class GreetingNode:
    """
    Greeting handler node
    
    Responds to greetings and small talk
    """
    
    async def respond(self, state: ConversationState) -> ConversationState:
        """
        Generate greeting response
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with greeting
        """
        language = state.get("language", "en")
        
        # Get greeting response for language
        response = GREETING_RESPONSES.get(language, GREETING_RESPONSES["en"])
        
        state["reply"] = response
        state["source"] = "greeting"
        state["context_used"] = False
        state["metadata"] = {"type": "greeting"}
        
        logger.info("greeting_response_generated", language=language)
        
        return state


class EscalationNode:
    """
    Escalation handler node
    
    Handles unclear or out-of-scope queries
    """
    
    async def respond(self, state: ConversationState) -> ConversationState:
        """
        Generate escalation response
        
        This is Node 3 in the LangGraph workflow
        
        Args:
            state: Current conversation state
        
        Returns:
            Updated state with escalation message
        """
        language = state.get("language", "en")
        
        # Get escalation response for language
        response = ESCALATION_RESPONSES.get(language, ESCALATION_RESPONSES["en"])
        
        state["reply"] = response
        state["source"] = "escalation"
        state["context_used"] = False
        state["metadata"] = {
            "type": "escalation",
            "reason": "unclear_intent"
        }
        
        logger.info("escalation_response_generated", language=language)
        
        return state


class ConversationalWorkflow:
    """
    LangGraph-based conversational workflow
    
    Workflow:
    1. Classifier Node → Classifies intent
    2. Conditional Routing → Routes to appropriate handler
    3. Handler Nodes → Generate response (RAG/Greeting/Escalation)
    """
    
    def __init__(self):
        """Initialize workflow"""
        self.classifier = IntentClassifier()
        self.rag_responder = RAGResponderNode()
        self.greeting_handler = GreetingNode()
        self.escalation_handler = EscalationNode()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("conversational_workflow_initialized")
    
    def _route_after_classification(
        self,
        state: ConversationState
    ) -> Literal["rag_responder", "greeting", "escalation"]:
        """
        Route to next node based on classification
        
        Args:
            state: Current state
        
        Returns:
            Next node name
        """
        category = state["category"]
        confidence = state["confidence"]
        
        # Route based on category
        if category == "greeting":
            logger.debug("routing_to_greeting")
            return "greeting"
        elif category in ["products", "returns", "general"]:
            logger.debug("routing_to_rag", category=category)
            return "rag_responder"
        else:
            logger.debug("routing_to_escalation", category=category)
            return "escalation"
    
    def _build_workflow(self) -> StateGraph:
        """
        Build LangGraph workflow
        
        Returns:
            Compiled workflow graph
        """
        # Create graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classifier", self.classifier.classify)
        workflow.add_node("rag_responder", self.rag_responder.respond)
        workflow.add_node("greeting", self.greeting_handler.respond)
        workflow.add_node("escalation", self.escalation_handler.respond)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Add conditional routing from classifier
        workflow.add_conditional_edges(
            "classifier",
            self._route_after_classification,
            {
                "rag_responder": "rag_responder",
                "greeting": "greeting",
                "escalation": "escalation"
            }
        )
        
        # All handler nodes end the workflow
        workflow.add_edge("rag_responder", END)
        workflow.add_edge("greeting", END)
        workflow.add_edge("escalation", END)
        
        # Compile the graph
        return workflow.compile()
    
    async def execute(
        self,
        message: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow for a message
        
        Args:
            message: User message
            session_id: Session ID
            conversation_history: Previous conversation
        
        Returns:
            Workflow result with response
        """
        start_time = datetime.now()
        
        # Initialize state
        initial_state: ConversationState = {
            "message": message,
            "session_id": session_id,
            "conversation_history": conversation_history or [],
            "category": "",
            "confidence": 0.0,
            "entities": {},
            "language": "en",
            "retrieved_chunks": [],
            "context_used": False,
            "reply": "",
            "source": "",
            "metadata": {}
        }
        
        logger.info(
            "workflow_execution_started",
            session_id=session_id,
            message_length=len(message)
        )
        
        try:
            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build response
            response = {
                "session_id": session_id,
                "category": result["category"],
                "reply": result["reply"],
                "confidence": result["confidence"],
                "source": result["source"],
                "context_used": result["context_used"],
                "language": result.get("language", "en"),
                "entities": result.get("entities", {}),
                "metadata": result.get("metadata", {}),
                "execution_time_ms": int(execution_time),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                "workflow_execution_completed",
                session_id=session_id,
                category=result["category"],
                source=result["source"],
                execution_time_ms=int(execution_time)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "workflow_execution_failed",
                session_id=session_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            # Return error response
            return {
                "session_id": session_id,
                "category": "error",
                "reply": "I apologize, but I encountered an error processing your request. Please try again.",
                "confidence": 0.0,
                "source": "error",
                "context_used": False,
                "metadata": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }