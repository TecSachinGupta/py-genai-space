# AI Data Platform

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
<!-- ![Tests](https://img.shields.io/badge/tests-passing-brightgreen) -->

**A universal platform for building AI and data engineering applications**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Contributing](#contributing)

</div>

---

## 🌟 Overview

AI Data Platform is a **generic, technology-agnostic framework** that unifies AI and data engineering workflows. Build chatbots, search systems, reasoning agents, data pipelines, and more—all with a single, coherent architecture.

### Why This Platform?

- **🎯 Generic by Design** - No AI-specific naming; works for any application
- **🔄 Unified Architecture** - Data engineering and AI share the same infrastructure
- **🧩 Modular Components** - Mix and match providers, storage, and processing
- **🚀 Production Ready** - Security, monitoring, and deployment built-in
- **📚 Well Documented** - Comprehensive guides and examples
- **🔌 Extensible** - Plugin architecture for custom components

---

## ✨ Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multiple Providers** | OpenAI, Anthropic, Google, Cohere, local models |
| **Vector Storage** | FAISS, Chroma, Qdrant, Pinecone, Weaviate |
| **Data Sources** | Files, databases, APIs, documents, web scraping |
| **Retrieval Systems** | Semantic, lexical, hybrid search with reranking |
| **Execution Engines** | Sequential, parallel, conditional, reactive workflows |
| **Reasoning Strategies** | Chain-of-thought, ReAct, tree-of-thought, planning |
| **Memory Systems** | Buffer, sliding window, semantic, entity tracking |
| **Interaction Protocols** | MCP, OpenAI-compatible API, REST, WebSocket |
| **Monitoring** | Metrics, structured logging, distributed tracing |
| **Security** | Authentication, authorization, input validation |

### Built-in Applications

- 💬 **Chat Application** - Conversational AI with memory
- 🔍 **Search System** - Semantic document search
- 🤖 **Reasoning Assistant** - Tool-using AI agent
- 🌐 **API Service** - REST API with OpenAPI docs
- 🔌 **Protocol Server** - MCP server implementation

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/TecSachinGupta/py-genai-space.git
cd py-genai-space

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

### 5-Minute Example: Simple Completion

```python
from src.providers.completion import OpenAIProvider

# Initialize provider
provider = OpenAIProvider(model="gpt-4")

# Generate completion
response = provider.complete("Explain quantum computing in simple terms")
print(response)
```

### 10-Minute Example: Document Search

```python
from src.data.sources import DocumentSource
from src.storage.indexes.vector import ChromaStore
from src.retrieval.strategies import SemanticRetrieval
from src.providers.embedding import OpenAIEmbedding

# Load and index documents
docs = DocumentSource().load("./documents/*.pdf")
embedder = OpenAIEmbedding(model="text-embedding-3-small")
store = ChromaStore(collection="docs")
store.index(docs, embedder)

# Search
retriever = SemanticRetrieval(store=store, top_k=5)
results = retriever.search("What is machine learning?")

for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Content: {result.content[:200]}...")
    print()
```

### 15-Minute Example: Reasoning Agent

```python
from src.reasoning.strategies import ReactStrategy
from src.reasoning.tools import SearchTool, CalculatorTool
from src.providers.completion import OpenAIProvider
from src.execution.engines import SequentialEngine

# Setup tools
tools = [
    SearchTool(api_key="your-search-api-key"),
    CalculatorTool()
]

# Create reasoning agent
provider = OpenAIProvider(model="gpt-4")
strategy = ReactStrategy(provider=provider, tools=tools)
engine = SequentialEngine(strategy=strategy)

# Execute reasoning
result = engine.run("What's 15% of the current population of Tokyo?")
print(result.answer)
print("\nReasoning steps:")
for step in result.reasoning_trace:
    print(f"  {step}")
```

---

## 📁 Project Structure

```
ai-data-platform/
├── config/                 # Configuration management
├── src/
│   ├── core/              # Base abstractions & interfaces
│   ├── providers/         # LLM, embedding, multimodal providers
│   ├── data/              # Data loading, processing, transformation
│   ├── storage/           # Databases, vectors, cache, objects
│   ├── retrieval/         # Search strategies & ranking
│   ├── execution/         # Task execution & orchestration
│   ├── reasoning/         # Decision-making strategies
│   ├── context/           # Memory & state management
│   ├── interaction/       # User interfaces & protocols
│   ├── pipelines/         # End-to-end workflows
│   ├── monitoring/        # Metrics, logging, tracing
│   ├── evaluation/        # Quality metrics & benchmarks
│   ├── security/          # Auth, validation, encryption
│   └── utilities/         # Helper functions
├── applications/          # Ready-to-use applications
├── workflows/             # Airflow/Prefect DAGs
├── examples/              # Usage examples
├── notebooks/             # Jupyter notebooks
├── deployments/           # Docker, K8s, cloud configs
├── docs/                  # Documentation
├── scripts/               # Utility scripts
└── tests/                 # Test suite
```

See [STRUCTURE_GUIDE.md](STRUCTURE_GUIDE.md) for detailed architecture.

---

## 📚 Documentation

### Getting Started
- [Installation Guide](docs/getting-started/installation.md)
- [Quick Start Tutorial](docs/getting-started/quickstart.md)
- [Configuration](docs/getting-started/configuration.md)
- [First Application](docs/getting-started/first-application.md)

### Core Concepts
- [Architecture Overview](docs/core-concepts/architecture.md)
- [Providers System](docs/core-concepts/providers.md)
- [Storage Layer](docs/core-concepts/storage.md)
- [Retrieval Systems](docs/core-concepts/retrieval.md)
- [Execution Engine](docs/core-concepts/execution.md)
- [Reasoning Strategies](docs/core-concepts/reasoning.md)

### User Guides
- [Building Chat Applications](docs/guides/chat-applications.md)
- [Creating Search Systems](docs/guides/search-systems.md)
- [Developing Reasoning Agents](docs/guides/reasoning-systems.md)
- [Data Pipeline Integration](docs/guides/data-pipelines.md)
- [Monitoring & Observability](docs/guides/monitoring.md)
- [Security Best Practices](docs/guides/security.md)

### API Reference
- [Core API](docs/api-reference/core.md)
- [Providers API](docs/api-reference/providers.md)
- [Storage API](docs/api-reference/storage.md)
- [Retrieval API](docs/api-reference/retrieval.md)

---

## 💡 Examples

### Chat with Memory

```python
from src.interaction.conversations import ConversationManager
from src.providers.completion import AnthropicProvider
from src.context.memory import BufferMemory

provider = AnthropicProvider(model="claude-3-5-sonnet")
memory = BufferMemory(max_messages=10)
chat = ConversationManager(provider=provider, memory=memory)

# Multi-turn conversation
response1 = chat.send_message("My name is Alice")
response2 = chat.send_message("What's my name?")  # Remembers "Alice"
```

### Hybrid Search with Reranking

```python
from src.retrieval.strategies import HybridRetrieval
from src.retrieval.ranking import CrossEncoderReranker

retriever = HybridRetrieval(
    vector_store=vector_store,
    vector_weight=0.7,
    lexical_weight=0.3
)

reranker = CrossEncoderReranker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

results = retriever.search("machine learning basics", top_k=10)
reranked = reranker.rerank(results, top_k=5)
```

### Data Pipeline Integration

```python
from src.pipelines import CompositePipeline, IngestionPipeline, InferencePipeline
from src.data.sources import DatabaseSource

pipeline = CompositePipeline(
    stages=[
        # Stage 1: Load from database
        IngestionPipeline(
            source=DatabaseSource("postgresql://..."),
            destination=vector_store
        ),
        # Stage 2: AI inference
        InferencePipeline(
            provider=OpenAIProvider(model="gpt-4")
        )
    ]
)

result = pipeline.run(query="Analyze customer sentiment")
```

### Custom Tool for Reasoning

```python
from src.reasoning.tools import BaseTool

class DatabaseQueryTool(BaseTool):
    """Tool that queries a database"""
    
    name = "database_query"
    description = "Query the database using SQL"
    
    def execute(self, query: str) -> str:
        # Execute SQL query
        result = self.db.execute(query)
        return str(result)

# Use in reasoning agent
tools = [DatabaseQueryTool(db=your_db), SearchTool(), CalculatorTool()]
agent = ReactStrategy(provider=provider, tools=tools)
```

More examples in [examples/](examples/) directory.

---

## 🏗️ Architecture Highlights

### Generic Design Philosophy

Unlike frameworks tied to specific AI use cases, this platform uses **generic terminology**:

| Generic Term | Applies To |
|--------------|------------|
| `providers` | LLMs, embeddings, databases, APIs, cloud services |
| `retrieval` | Semantic search, SQL queries, API calls, web scraping |
| `execution` | AI reasoning, ETL jobs, business logic, workflows |
| `reasoning` | AI decision-making, rule engines, optimization |
| `context` | Conversation history, session state, application state |
| `interaction` | Chat interfaces, APIs, protocols, webhooks |

### Unified Data Flow

```
┌─────────────┐
│ Data Source │ (Files, DBs, APIs, Web)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Processor  │ (Clean, Transform, Validate)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Storage   │ (Vector, Relational, Cache)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │ (Search, Query, Filter)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reasoning  │ (Analyze, Decide, Plan)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Execution  │ (Run Tasks, Call Tools)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Interaction │ (Return Results)
└─────────────┘
```

---

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Application
ENVIRONMENT=development
DEBUG=True

# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...

# Storage
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379/0
VECTOR_STORE_TYPE=chroma

# Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
BATCH_SIZE=32

# Monitoring
ENABLE_METRICS=True
LOG_LEVEL=INFO
```

### Configuration Files

- `config/settings.py` - Main application settings (Pydantic)
- `config/providers.yaml` - Provider registry and configurations
- `config/pipelines.yaml` - Pipeline configurations
- `config/logging.yaml` - Logging configuration

See [Configuration Guide](docs/getting-started/configuration.md) for details.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/test_providers/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/
```

---

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t ai-data-platform .

# Run container
docker run -p 8000:8000 --env-file .env ai-data-platform

# Use docker-compose
docker-compose up
```

### Kubernetes

```bash
# Apply configurations
kubectl apply -f deployments/kubernetes/base/

# Deploy to specific environment
kubectl apply -k deployments/kubernetes/overlays/production/
```

### Cloud Providers

- **AWS**: See [deployments/cloud/aws/](deployments/cloud/aws/)
- **GCP**: See [deployments/cloud/gcp/](deployments/cloud/gcp/)
- **Azure**: See [deployments/cloud/azure/](deployments/cloud/azure/)

---

## 📊 Monitoring & Observability

### Metrics

```python
from src.monitoring.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record("request_duration", 0.523, labels={"endpoint": "/api/completion"})
metrics.record("tokens_used", 1500, labels={"model": "gpt-4"})

# Get statistics
stats = metrics.get_stats("request_duration")
print(f"Average: {stats['mean']:.3f}s")
```

### Structured Logging

```python
from src.monitoring.logging import StructuredLogger

logger = StructuredLogger("app")
request_logger = logger.bind(request_id="req-123", user_id="user-456")
request_logger.info("Processing request", endpoint="/api/completion")
```

### Distributed Tracing

```python
from src.monitoring.tracing import Tracer

tracer = Tracer()

with tracer.start_span("process_request") as span:
    span.set_attribute("user_id", "user-123")
    
    with tracer.start_span("retrieve_documents"):
        # Retrieval logic
        pass
    
    with tracer.start_span("generate_response"):
        # Generation logic
        pass
```

---

## 🔒 Security

- **Authentication**: JWT, OAuth2, API keys, mTLS
- **Authorization**: RBAC, ABAC, policy engine
- **Input Validation**: SQL injection, prompt injection detection
- **Encryption**: Data at rest and in transit
- **Secret Management**: Integration with Vault, AWS Secrets Manager

See [Security Guide](docs/guides/security.md) for best practices.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/TecSachinGupta/py-genai-space.git
cd py-genai-space

# Setup development environment
bash scripts/setup/init_dev_environment.sh

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

### Code Style

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pylint** for linting

```bash
# Format code
black src/
isort src/

# Type check
mypy src/

# Lint
pylint src/
```

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## 🙏 Acknowledgments

Built with:
- [OpenAI](https://openai.com/) - GPT models
- [Anthropic](https://anthropic.com/) - Claude models
- [LangChain](https://langchain.com/) - Inspiration for abstractions
- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Pydantic](https://pydantic.dev/) - Data validation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Issue Tracker**: [GitHub Issues](https://github.com/TecSachinGupta/py-genai-space/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TecSachinGupta/py-genai-space/discussions)

---

## 💬 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/...)
- 📖 Documentation: [Read the docs](docs/)
- 🐛 Report bugs: [GitHub Issues](https://github.com/TecSachinGupta/py-genai-space/issues)

---

<div align="center">

**[⬆ back to top](#ai-data-platform)**

Made with ❤️ by the AI Data Platform team

</div>