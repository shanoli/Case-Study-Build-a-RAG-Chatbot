# RAG Chatbot - Production Ready Implementation

## 🎯 Overview

A production-ready conversational chatbot with:
- ✅ **Vector Search** (ChromaDB + text-embedding-004)
- ✅ **Multi-lingual Support** (EN, HI, TA, TE, ES, FR)
- ✅ **RAG Pipeline** (Semantic Retrieval + Generation)
- ✅ **Gemini Integration** (Latest models)

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your GOOGLE_API_KEY to .env

# 3. Test Core Components
python -c "
import asyncio
from src.services.knowledge_base import KnowledgeBaseService
from src.services.rag_engine import RAGEngine

async def test():
    # Initialize KB
    kb = KnowledgeBaseService()
    await kb.initialize()
    
    # Upload knowledge base
    with open('data/product_info.txt', 'r') as f:
        content = f.read()
    result = await kb.upload_knowledge_base(content)
    print(f'✅ Uploaded: {result}')
    
    # Test RAG
    rag = RAGEngine()
    result = await rag.rag_query('What is the price of SmartWatch?')
    print(f'✅ Answer: {result[\"generated_text\"]}')

asyncio.run(test())
"
```

## 🔍 Key Features

### Vector Search
- **Model**: text-embedding-004 (768 dimensions)
- **Method**: Cosine similarity
- **Threshold**: 0.7 (configurable)
- **Index**: HNSW for fast approximate search

### Multi-Lingual
- **Supported**: English, Hindi, Tamil, Telugu, Spanish, French
- **Detection**: Automatic language detection
- **Response**: Same language as query

### Knowledge Base
- **Source**: product_info.txt (TechGear products)
- **Chunks**: Semantic chunking (500 chars, 50 overlap)
- **Storage**: ChromaDB with persistence

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── core/          # Configuration, logging, exceptions
│   ├── services/      # KB, RAG, Sessions, LangGraph
│   ├── utils/         # Prompts, helpers
│   └── api/           # FastAPI routes (TODO)
├── data/              # Knowledge base files
├── tests/             # Unit and integration tests
└── requirements.txt   # Dependencies
```

## 🧪 Testing

See IMPLEMENTATION_COMPLETE.md for:
- Vector search examples
- Multi-lingual queries
- API endpoints (after FastAPI implementation)

## 📊 Architecture

See VECTOR_SEARCH_FLOW.mermaid for detailed flow diagram.

## 🔧 Configuration

All settings in `.env`:
- `GOOGLE_API_KEY` - Required
- `GEMINI_MODEL` - Default: gemini-1.5-pro-latest
- `EMBEDDING_MODEL` - Default: models/text-embedding-004
- `RELEVANCE_THRESHOLD` - Default: 0.7

## 📝 License

MIT
