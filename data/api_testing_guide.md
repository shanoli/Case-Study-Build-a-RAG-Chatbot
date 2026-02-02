# API Testing Guide - RAG Chatbot

## Complete Test Suite for All Tasks

This guide provides comprehensive testing commands for each task and API endpoint.

---

## Task 1: Knowledge Base Setup Tests

### Test 1.1: Upload Knowledge Base
```bash
# Basic upload
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@knowledge_base.txt" \
  -F "chunk_size=500" \
  -F "chunk_overlap=50" \
  -F "replace_existing=false"

# Expected Response:
{
  "status": "success",
  "chunks_created": 245,
  "embeddings_generated": 245,
  "duplicates_skipped": 0,
  "processing_time_ms": 12340,
  "model_used": "models/embedding-001"
}
```

### Test 1.2: Check Knowledge Base Status
```bash
curl -X GET http://localhost:8000/knowledge/status

# Expected Response:
{
  "total_chunks": 245,
  "total_documents": 1,
  "embedding_model": "models/embedding-001",
  "last_updated": "2026-01-30T09:00:00Z",
  "storage_size_mb": 12.3,
  "index_ready": true
}
```

### Test 1.3: Inspect Specific Embedding
```bash
curl -X GET http://localhost:8000/debug/embeddings/chunk_42

# Expected Response:
{
  "chunk_id": "chunk_42",
  "text": "Our return policy allows customers to return...",
  "embedding_dimension": 768,
  "embedding_preview": [0.023, -0.045, 0.189, "... (truncated)"],
  "metadata": {
    "source": "knowledge_base.txt",
    "created_at": "2026-01-30T09:00:00Z",
    "chunk_index": 42
  }
}
```

### Test 1.4: Test Duplicate Prevention
```bash
# Upload same file again
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@knowledge_base.txt" \
  -F "replace_existing=false"

# Expected Response:
{
  "status": "success",
  "chunks_created": 0,
  "embeddings_generated": 0,
  "duplicates_skipped": 245,
  "message": "All chunks already exist in database"
}
```

### Test 1.5: Clear Knowledge Base
```bash
curl -X DELETE http://localhost:8000/knowledge/clear

# Expected Response:
{
  "status": "cleared",
  "chunks_deleted": 245,
  "timestamp": "2026-01-30T10:30:00Z"
}
```

---

## Task 2: RAG Chain Tests

### Test 2.1: Retrieval Only
```bash
curl -X POST http://localhost:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your return policy?",
    "top_k": 5
  }'

# Expected Response:
{
  "query": "What is your return policy?",
  "chunks": [
    {
      "content": "Our return policy allows customers to return items within 30 days...",
      "score": 0.89,
      "metadata": {
        "source": "knowledge_base.txt",
        "chunk_id": "chunk_42"
      }
    },
    {
      "content": "For returns, please ensure the item is in original condition...",
      "score": 0.82,
      "metadata": {
        "source": "knowledge_base.txt",
        "chunk_id": "chunk_43"
      }
    }
  ],
  "total_chunks": 5,
  "avg_score": 0.76,
  "retrieval_time_ms": 45
}
```

### Test 2.2: Test with No Relevant Context
```bash
curl -X POST http://localhost:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the weather like?",
    "top_k": 3
  }'

# Expected Response:
{
  "query": "What is the weather like?",
  "chunks": [],
  "total_chunks": 0,
  "avg_score": 0.0,
  "retrieval_time_ms": 23,
  "message": "No relevant context found (all scores below 0.7 threshold)"
}
```

### Test 2.3: RAG Generation
```bash
curl -X POST http://localhost:8000/debug/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your return policy?",
    "context": "Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached.",
    "use_rag": true
  }'

# Expected Response:
{
  "generated_text": "Based on our policy, you can return items within 30 days of purchase as long as they remain in their original condition with all tags attached.",
  "model": "gemini-1.5-pro",
  "tokens_used": 156,
  "latency_ms": 823,
  "context_used": true
}
```

### Test 2.4: Test Error Handling (No Context)
```bash
curl -X POST http://localhost:8000/debug/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the weather?",
    "context": "",
    "use_rag": false
  }'

# Expected Response:
{
  "generated_text": "I don't have information about that in our knowledge base. Could you ask something related to our products or policies?",
  "model": "gemini-1.5-pro",
  "tokens_used": 45,
  "latency_ms": 412,
  "context_used": false,
  "fallback": true
}
```

---

## Task 3: LangGraph Workflow Tests

### Test 3.1: Classifier - Products Query
```bash
curl -X POST http://localhost:8000/debug/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What products do you offer?"
  }'

# Expected Response:
{
  "category": "products",
  "confidence": 0.92,
  "reasoning": "Query contains product-related keywords: 'products', 'offer'",
  "alternative_categories": {
    "general": 0.08
  },
  "keywords_matched": ["products", "offer"],
  "processing_time_ms": 12
}
```

### Test 3.2: Classifier - Returns Query
```bash
curl -X POST http://localhost:8000/debug/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I return an item?"
  }'

# Expected Response:
{
  "category": "returns",
  "confidence": 0.95,
  "reasoning": "Query matches return intent pattern",
  "alternative_categories": {
    "general": 0.05
  },
  "keywords_matched": ["return", "item"]
}
```

### Test 3.3: Classifier - Greeting
```bash
curl -X POST http://localhost:8000/debug/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!"
  }'

# Expected Response:
{
  "category": "greeting",
  "confidence": 1.0,
  "reasoning": "Message matches greeting pattern",
  "alternative_categories": {},
  "keywords_matched": ["hello"]
}
```

### Test 3.4: Classifier - Unknown Intent
```bash
curl -X POST http://localhost:8000/debug/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "asdfghjkl random text"
  }'

# Expected Response:
{
  "category": "unknown",
  "confidence": 0.1,
  "reasoning": "No matching patterns found",
  "alternative_categories": {
    "general": 0.1
  },
  "keywords_matched": []
}
```

---

## Task 4: Chat Endpoint Tests

### Test 4.1: First Message (New Session)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!"
  }'

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "category": "greeting",
  "reply": "Hello! Welcome to our customer support. I can help you with questions about our products, returns, and general inquiries. How can I assist you today?",
  "confidence": "high",
  "source": "greeting",
  "context_used": false,
  "retrieved_chunks": 0,
  "timestamp": "2026-01-30T10:30:00Z",
  "new_session": true
}
```

### Test 4.2: Product Query with Session
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "What products do you sell?"
  }'

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "category": "products",
  "reply": "We offer a wide range of electronics including smartphones, laptops, tablets, and accessories. Our catalog includes top brands like...",
  "confidence": "high",
  "source": "rag",
  "context_used": true,
  "retrieved_chunks": 3,
  "timestamp": "2026-01-30T10:31:15Z"
}
```

### Test 4.3: Return Policy Query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "What is your return policy?"
  }'

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "category": "returns",
  "reply": "Our return policy allows you to return items within 30 days of purchase. The item must be in its original condition with all tags and packaging intact. You'll receive a full refund to your original payment method within 5-7 business days.",
  "confidence": "high",
  "source": "rag",
  "context_used": true,
  "retrieved_chunks": 4,
  "timestamp": "2026-01-30T10:32:30Z"
}
```

### Test 4.4: Unknown Query (Escalation)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "Can you help me file my taxes?"
  }'

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "category": "unknown",
  "reply": "I'm sorry, but I'm specifically designed to help with product inquiries and return policies. For tax-related questions, I'd recommend consulting with a tax professional. Is there anything else about our products or services I can help you with?",
  "confidence": "low",
  "source": "escalation",
  "context_used": false,
  "retrieved_chunks": 0,
  "timestamp": "2026-01-30T10:33:45Z",
  "escalation": true
}
```

### Test 4.5: Contextual Follow-up
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "message": "What if the item is damaged?"
  }'

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "category": "returns",
  "reply": "If you received a damaged item, you can return it at any time - our 30-day limit doesn't apply to damaged goods. Please contact our support team with photos of the damage, and we'll arrange a replacement or full refund immediately.",
  "confidence": "high",
  "source": "rag",
  "context_used": true,
  "retrieved_chunks": 3,
  "timestamp": "2026-01-30T10:34:20Z",
  "conversation_context_used": true
}
```

---

## Session Management Tests

### Test 5.1: Get Session Details
```bash
curl -X GET http://localhost:8000/sessions/550e8400-e29b-41d4-a716-446655440000

# Expected Response:
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-01-30T10:30:00Z",
  "last_active": "2026-01-30T10:34:20Z",
  "message_count": 5,
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello!",
      "timestamp": "2026-01-30T10:30:00Z",
      "category": "greeting"
    },
    {
      "role": "assistant",
      "content": "Hello! Welcome to our customer support...",
      "timestamp": "2026-01-30T10:30:01Z"
    },
    {
      "role": "user",
      "content": "What products do you sell?",
      "timestamp": "2026-01-30T10:31:15Z",
      "category": "products"
    }
  ],
  "metadata": {
    "user_agent": "curl/7.68.0",
    "ip_address": "127.0.0.1"
  }
}
```

### Test 5.2: List Active Sessions
```bash
curl -X GET http://localhost:8000/sessions/active

# Expected Response:
{
  "active_sessions": 3,
  "sessions": [
    {
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "created_at": "2026-01-30T10:30:00Z",
      "last_active": "2026-01-30T10:34:20Z",
      "message_count": 5
    },
    {
      "session_id": "660f9511-f39c-52e5-b827-557766551111",
      "created_at": "2026-01-30T10:25:00Z",
      "last_active": "2026-01-30T10:35:00Z",
      "message_count": 3
    }
  ],
  "timestamp": "2026-01-30T10:35:30Z"
}
```

### Test 5.3: Delete Session
```bash
curl -X DELETE http://localhost:8000/sessions/550e8400-e29b-41d4-a716-446655440000

# Expected Response:
{
  "status": "deleted",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-30T10:36:00Z"
}
```

---

## Health Check Tests

### Test 6.1: Basic Health Check
```bash
curl -X GET http://localhost:8000/health

# Expected Response (Healthy):
{
  "status": "healthy",
  "components": {
    "database": "connected",
    "llm": "available",
    "session_store": "active",
    "embedding_service": "operational"
  },
  "version": "1.0.0",
  "timestamp": "2026-01-30T10:30:00Z",
  "uptime_seconds": 3600
}

# Expected Response (Unhealthy):
{
  "status": "unhealthy",
  "components": {
    "database": "disconnected",
    "llm": "unavailable",
    "session_store": "active",
    "embedding_service": "operational"
  },
  "errors": [
    "ChromaDB connection failed",
    "Gemini API unreachable"
  ],
  "timestamp": "2026-01-30T10:30:00Z"
}
```

---

## Error Handling Tests

### Test 7.1: Invalid Session ID
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "invalid-session-id",
    "message": "Hello"
  }'

# Expected Response:
{
  "error": "invalid_session",
  "message": "Session not found. A new session has been created.",
  "session_id": "770g0622-g40d-63f6-c938-668877662222",
  "category": "greeting",
  "reply": "Hello! Welcome back. How can I help you today?",
  "timestamp": "2026-01-30T10:37:00Z"
}
```

### Test 7.2: Empty Message
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": ""
  }'

# Expected Response (400 Bad Request):
{
  "error": "validation_error",
  "message": "Message cannot be empty",
  "timestamp": "2026-01-30T10:38:00Z"
}
```

### Test 7.3: Message Too Long
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "' + 'a' * 5000 + '"
  }'

# Expected Response (400 Bad Request):
{
  "error": "validation_error",
  "message": "Message exceeds maximum length of 2000 characters",
  "timestamp": "2026-01-30T10:39:00Z"
}
```

---

## Performance Testing

### Test 8.1: Concurrent Requests
```bash
# Using Apache Bench
ab -n 100 -c 10 -p chat_payload.json \
  -T application/json \
  http://localhost:8000/chat

# Expected Output:
Requests per second:    45.23 [#/sec]
Time per request:       221.1 [ms] (mean)
Time per request:       22.1 [ms] (mean, across all concurrent requests)
```

### Test 8.2: Stress Test Knowledge Upload
```bash
# Upload large file
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@large_knowledge_base.txt" \
  -F "chunk_size=1000"

# Monitor processing time in response
```

---

## Integration Test Scenarios

### Scenario 1: Complete User Journey
```bash
# 1. New user greeting
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi"}' | jq -r '.session_id')

# 2. Ask about products
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"What laptops do you have?\"}"

# 3. Ask about returns
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"Can I return it if I don't like it?\"}"

# 4. Get session history
curl -X GET http://localhost:8000/sessions/$SESSION
```

### Scenario 2: Knowledge Base Update Flow
```bash
# 1. Check current status
curl -X GET http://localhost:8000/knowledge/status

# 2. Clear existing data
curl -X DELETE http://localhost:8000/knowledge/clear

# 3. Upload new knowledge base
curl -X POST http://localhost:8000/knowledge/upload \
  -F "file=@updated_kb.txt"

# 4. Verify new status
curl -X GET http://localhost:8000/knowledge/status

# 5. Test retrieval with new data
curl -X POST http://localhost:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "new product feature", "top_k": 3}'
```

---

## Automated Test Script

```bash
#!/bin/bash

# RAG Chatbot API Test Suite
BASE_URL="http://localhost:8000"

echo "🧪 Running RAG Chatbot Test Suite"
echo "=================================="

# Test 1: Health Check
echo "
Test 1: Health Check"
curl -s $BASE_URL/health | jq .

# Test 2: Upload Knowledge Base
echo "
Test 2: Upload Knowledge Base"
curl -s -X POST $BASE_URL/knowledge/upload \
  -F "file=@knowledge_base.txt" | jq .

# Test 3: Test Classifier
echo "
Test 3: Test Classifier"
curl -s -X POST $BASE_URL/debug/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your products?"}' | jq .

# Test 4: Test Retrieval
echo "
Test 4: Test Retrieval"
curl -s -X POST $BASE_URL/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "return policy", "top_k": 3}' | jq .

# Test 5: Chat Flow
echo "
Test 5: Chat Flow"
SESSION=$(curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' | jq -r '.session_id')

echo "Session created: $SESSION"

curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"What is your return policy?\"}" | jq .

echo "
✅ All tests completed!"
```

---

## Postman Collection

Save this as `rag_chatbot.postman_collection.json`:

```json
{
  "info": {
    "name": "RAG Chatbot API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/health"
      }
    },
    {
      "name": "Upload Knowledge Base",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/knowledge/upload",
        "body": {
          "mode": "formdata",
          "formdata": [
            {
              "key": "file",
              "type": "file",
              "src": "knowledge_base.txt"
            }
          ]
        }
      }
    },
    {
      "name": "Chat - New Session",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/chat",
        "body": {
          "mode": "raw",
          "raw": "{\n  \"message\": \"Hello!\"\n}"
        }
      }
    }
  ],
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000"
    }
  ]
}
```

---

## Summary

This testing guide covers:

✅ **Task 1**: Knowledge base upload, status, embeddings inspection  
✅ **Task 2**: RAG retrieval and generation testing  
✅ **Task 3**: Classifier and workflow node testing  
✅ **Task 4**: Chat API with session management  
✅ **Additional**: Health checks, error handling, performance tests  

Each test includes:
- Command example
- Expected response
- Error scenarios
- Integration workflows
