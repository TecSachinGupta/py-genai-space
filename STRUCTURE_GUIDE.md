# AI Data Platform - Structure Guide

**Complete architectural guide for understanding and extending the platform**

---

## 📋 Table of Contents

- [Design Philosophy](#design-philosophy)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Module Details](#module-details)
- [Data Flow Patterns](#data-flow-patterns)
- [Extension Guide](#extension-guide)
- [Best Practices](#best-practices)

---

## 🎯 Design Philosophy

### 1. Generic by Design

**Problem**: AI-specific naming limits reusability and creates confusion when integrating with traditional data engineering.

**Solution**: Use generic terminology that works for both AI and data engineering:

| ❌ Avoid | ✅ Use Instead | Why |
|---------|---------------|-----|
| `genai_models` | `providers` | Providers can be LLMs, databases, APIs, etc. |
| `llm_providers` | `providers/completion` | Generic providers with specific types |
| `rag_pipeline` | `retrieval` + `pipelines` | Retrieval is broader than RAG |
| `agent_tools` | `reasoning/tools` | Tools work for any reasoning system |
| `agent_memory` | `context/memory` | Context management is universal |
| `chatbot` | `interaction/conversations` | Interactions include APIs, protocols |

### 2. Separation of Concerns

Each module has a single, clear responsibility:

```
┌─────────────┐
│    core     │  Base abstractions & interfaces
└─────────────┘
       ▲
       │ implements
       │
┌─────────────┐
│  providers  │  External service integrations
└─────────────┘
       │
       ▼ uses
┌─────────────┐
│    data     │  Load, transform, validate data
└─────────────┘
       │
       ▼ stores in
┌─────────────┐
│   storage   │  Persist data (vectors, DBs, cache)
└─────────────┘
       │
       ▼ searches via
┌─────────────┐
│  retrieval  │  Find relevant information
└─────────────┘
       │
       ▼ reasons with
┌─────────────┐
│  reasoning  │  Make decisions
└─────────────┘
       │
       ▼ executes via
┌─────────────┐
│  execution  │  Run tasks
└─────────────┘
       │
       ▼ maintains
┌─────────────┐
│   context   │  Track state
└─────────────┘
       │
       ▼ interacts via
┌─────────────┐
│interaction  │  User/system communication
└─────────────┘
```

### 3. Composability

Components are designed to work together seamlessly:

```python
# Example: Compose a complete RAG system
from src.data.sources import DocumentSource
from src.data.transformers import Chunking
from src.providers.embedding import OpenAIEmbedding
from src.storage.indexes.vector import ChromaStore
from src.retrieval.strategies import HybridRetrieval
from src.providers.completion import OpenAIProvider

# Each component is independent but composable
pipeline = (
    DocumentSource()
    >> Chunking(strategy="semantic")
    >> OpenAIEmbedding()
    >> ChromaStore()
    >> HybridRetrieval()
    >> OpenAIProvider()
)
```

### 4. Plugin Architecture

Easily extend with custom components:

```python
from src.core.base import BaseComponent
from src.core.interfaces import CompletionProvider

class CustomProvider(BaseComponent, CompletionProvider):
    """Custom provider implementation"""
    
    def initialize(self):
        # Setup logic
        pass
    
    def complete(self, prompt: str, **kwargs) -> str:
        # Custom completion logic
        return response

# Register and use
from src.core.registry import ComponentRegistry
ComponentRegistry.register("custom", CustomProvider)
```

---

## 📁 Directory Structure

### Overview

#### Simple View
```
ai-data-platform/
├── config/                    # Configuration management
├── src/                       # Source code
├── applications/              # Ready-to-use applications
├── workflows/                 # Orchestration workflows
├── examples/                  # Usage examples
├── notebooks/                 # Jupyter notebooks
├── deployments/               # Deployment configs
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── tests/                     # Test suite
└── assets/                    # Static resources
```

#### Detailed View
```
📁 py-genai-space/
├── 📁 .github/
│   ├── 📁 ISSUE_TEMPLATE/
│   │   ├── 📝 bug_report.md
│   │   ├── 📝 feature_request.md
│   │   └── 📝 question.md
│   ├── 🔄 workflows/
│   │   ├── ⚙️ cd.yml
│   │   ├── ⚙️ ci.yml
│   │   └── ⚙️ security.yml
│   └── 📝 pull_request_template.md
├── 📱 applications/
│   ├── 📁 api/
│   │   ├── 📚 docs/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 middleware/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 routes/
│   │   │   └── 📄 .gitkeep
│   │   └── 🐍 __init__.py
│   ├── 📁 assistant/
│   │   ├── 📁 interfaces/
│   │   │   └── 📄 .gitkeep
│   │   └── 🐍 __init__.py
│   ├── 📁 chat/
│   │   ├── 📁 interfaces/
│   │   │   └── 📄 .gitkeep
│   │   └── 🐍 __init__.py
│   ├── 📁 search/
│   │   ├── 📁 interfaces/
│   │   │   └── 📄 .gitkeep
│   │   └── 🐍 __init__.py
│   ├── 📁 server/
│   │   ├── 📁 handlers/
│   │   │   └── 📄 .gitkeep
│   │   └── 🐍 __init__.py
│   └── 🐍 __init__.py
├── 🎨 assets/
│   ├── 📁 data/
│   │   ├── 📁 processed/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 raw/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 sample/
│   │   │   ├── 📝 README.md
│   │   │   └── 📄 sample_data.csv
│   │   └── 📁 schemas/
│   │       └── 📋 user_schema.json
│   ├── 📁 dependencies/
│   │   └── 📝 README.md
│   ├── 📁 images/
│   │   └── 📝 README.md
│   ├── 📁 knowledge/
│   │   ├── 📁 documents/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 graphs/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 indexes/
│   │   │   └── 📄 .gitkeep
│   │   └── 📁 metadata/
│   │       └── 📄 .gitkeep
│   ├── 📁 models/
│   │   ├── 📁 artifacts/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 checkpoints/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 configs/
│   │   │   └── 📄 .gitkeep
│   │   └── 📁 metadata/
│   │       └── 📄 .gitkeep
│   └── 📁 resources/
│       ├── 📁 policies/
│       │   └── 📄 .gitkeep
│       ├── 📁 prompts/
│       │   └── 📄 .gitkeep
│       ├── 📁 schemas/
│       │   └── 📄 .gitkeep
│       └── 🔄 workflows/
│           └── 📄 .gitkeep
├── ⚙️ config/
│   ├── 🐍 __init__.py
│   ├── 🐍 logging.py
│   └── 🐍 settings.py
├── 🚀 deployments/
│   ├── 📁 cloud/
│   │   ├── 📁 aws/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 azure/
│   │   │   └── 📄 .gitkeep
│   │   └── 📁 gcp/
│   │       └── 📄 .gitkeep
│   ├── 📁 docker/
│   │   └── 🔧 scripts/
│   │       └── 📄 .gitkeep
│   ├── 📁 kubernetes/
│   │   ├── 📁 base/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 helm/
│   │   │   └── 📁 templates/
│   │   │       └── 📄 .gitkeep
│   │   └── 📁 overlays/
│   │       ├── 📁 development/
│   │       │   └── 📄 .gitkeep
│   │       ├── 📁 production/
│   │       │   └── 📄 .gitkeep
│   │       └── 📁 staging/
│   │           └── 📄 .gitkeep
│   └── 📁 terraform/
│       └── 📁 modules/
│           ├── 📁 compute/
│           │   └── 📄 .gitkeep
│           ├── 📁 networking/
│           │   └── 📄 .gitkeep
│           └── 📁 storage/
│               └── 📄 .gitkeep
├── 📚 docs/
│   ├── 📁 api/
│   │   └── 📝 README.md
│   ├── 📁 api-reference/
│   │   └── 📄 .gitkeep
│   ├── 📁 core-concepts/
│   │   └── 📄 .gitkeep
│   ├── 📁 deployment/
│   │   └── 📄 .gitkeep
│   ├── 📁 getting-started/
│   │   └── 📄 .gitkeep
│   ├── 📁 guides/
│   │   └── 📝 security.md
│   └── 📁 tutorials/
│       ├── 📝 etl_pipeline_guide.md
│       └── 📝 getting_started.md
├── 💡 examples/
│   ├── 📁 advanced/
│   │   └── 📄 .gitkeep
│   ├── 📁 integration/
│   │   └── 📄 .gitkeep
│   ├── 📁 intermediate/
│   │   └── 📄 .gitkeep
│   └── 📁 quickstart/
│       ├── 📄 .gitkeep
│       └── 🐍 01_simple_completion.py
├── 📦 src/
│   ├── 📁 context/
│   │   ├── 📁 memory/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 persistence/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 session/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 core/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 base.py
│   │   ├── 🐍 exceptions.py
│   │   └── 🐍 interfaces.py
│   ├── 📁 data/
│   │   ├── 📁 processors/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 serializers/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 sources/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 base.py
│   │   │   ├── 🐍 docx_processor.py
│   │   │   ├── 🐍 pdf_loader.py
│   │   │   ├── 🐍 pdf_processor.py
│   │   │   └── 🐍 txt_processor.py
│   │   ├── 📁 transformers/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 chunking.py
│   │   │   ├── 🐍 fixed_size_chunker.py
│   │   │   ├── 🐍 sementaic_chunker.py
│   │   │   ├── 🐍 sentence_chunker.py
│   │   │   └── 🐍 token_chunker.py
│   │   ├── 📁 validators/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 evaluation/
│   │   ├── 📁 benchmarks/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 experiments/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 judges/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 metrics/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 execution/
│   │   ├── 📁 engines/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 orchestration/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 state/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 tasks/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 interaction/
│   │   ├── 📁 conversations/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 formatting/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 protocols/
│   │   │   ├── 📁 mcp/
│   │   │   │   ├── 📁 transports/
│   │   │   │   │   └── 🐍 __init__.py
│   │   │   │   └── 🐍 __init__.py
│   │   │   ├── 📁 openai/
│   │   │   │   └── 🐍 __init__.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 streaming/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 monitoring/
│   │   ├── 📁 alerting/
│   │   │   ├── 📁 channels/
│   │   │   │   └── 🐍 __init__.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 logging/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 metrics/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 tracing/
│   │   │   ├── 📁 exporters/
│   │   │   │   └── 🐍 __init__.py
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 notebooks/
│   │   ├── 📁 00_archived/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 FaissStore.py
│   │   │   └── 📝 README.md
│   │   └── 🐍 __init__.py
│   ├── 📁 pipelines/
│   │   └── 🐍 __init__.py
│   ├── 📁 providers/
│   │   ├── 📁 completion/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 embedding/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 base.py
│   │   │   ├── 🐍 cohere_embedding.py
│   │   │   ├── 🐍 embedding_cache.py
│   │   │   ├── 🐍 huggingface_embedding.py
│   │   │   ├── 🐍 manager.py
│   │   │   └── 🐍 openai_embedding.py
│   │   ├── 📁 multimodal/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 reasoning/
│   │   ├── 📁 planning/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 selection/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 strategies/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 tools/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 retrieval/
│   │   ├── 📁 filters/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 query/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 ranking/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 strategies/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 schemas/
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 data_models.py
│   │   └── 🐍 file_data_models.py
│   ├── 📁 security/
│   │   ├── 📁 authentication/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 authorization/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 encryption/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 secrets/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 validation/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 storage/
│   │   ├── 📁 cache/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 databases/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 indexes/
│   │   │   ├── 📁 fulltext/
│   │   │   │   └── 🐍 __init__.py
│   │   │   ├── 📁 vector/
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 base.py
│   │   │   │   ├── 🐍 chroma_store.py
│   │   │   │   ├── 🐍 faiss_store.py
│   │   │   │   └── 🐍 inmemory_store.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 objects/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 queues/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   ├── 📁 utilities/
│   │   ├── 📁 concurrency/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 helpers/
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 common.py
│   │   ├── 📁 network/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 serialization/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 text/
│   │   │   └── 🐍 __init__.py
│   │   └── 🐍 __init__.py
│   └── 🐍 __init__.py
├── 📁 templates/
│   ├── ⚙️ config_template.yaml
│   └── 🐍 job_template.py
├── 🧪 tests/
│   ├── 📁 fixtures/
│   │   ├── 📁 configs/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 data/
│   │   │   └── 📁 sample_documents/
│   │   │       └── 📄 .gitkeep
│   │   ├── 📁 mocks/
│   │   │   └── 📄 .gitkeep
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 sample_data.py
│   ├── 📁 integration/
│   │   ├── 📁 test_applications/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_pipelines/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_storage_integration/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_workflows/
│   │   │   └── 📄 .gitkeep
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 test_database.py
│   ├── 📁 performance/
│   │   └── 🐍 __init__.py
│   ├── 📁 security/
│   │   └── 🐍 __init__.py
│   ├── 📁 unit/
│   │   ├── 📁 test_context/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_core/
│   │   │   ├── 📄 .gitkeep
│   │   │   └── 🐍 test_base.py
│   │   ├── 📁 test_data/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_execution/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_interaction/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_providers/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_reasoning/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_retrieval/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_storage/
│   │   │   └── 📄 .gitkeep
│   │   ├── 📁 test_utilities/
│   │   │   └── 📄 .gitkeep
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 test_sample.py
│   └── 🐍 __init__.py
├── 🔄 workflows/
│   ├── 📁 airflow/
│   │   ├── 📁 dags/
│   │   │   └── 📄 .gitkeep
│   │   └── 📁 operators/
│   │       └── 📄 .gitkeep
│   ├── 📁 prefect/
│   │   ├── 📁 flows/
│   │   │   └── 📄 .gitkeep
│   │   └── 📁 tasks/
│   │       └── 📄 .gitkeep
│   └── 📁 temporal/
│       ├── 📁 activities/
│       │   └── 📄 .gitkeep
│       └── 🔄 workflows/
│           └── 📄 .gitkeep
├── 📄 .env.example
├── 🚫 .gitignore
├── ⚙️ .pre-commit-config.yaml
├── 📝 CHANGELOG.md
├── 📝 CONTRIBUTING.md
├── 📜 LICENSE
├── ⚡ Makefile
├── ⚙️ mkdocs.yml
├── 📋 pyproject.toml
├── 📄 pytest.ini
├── 📝 README.md
├── 📄 requirements-dev.txt
├── 📄 requirements.txt
├── 🐍 setup.py
└── 📝 STRUCTURE_GUIDE.md
```

### Detailed Breakdown

#### 📦 config/ - Configuration Management

```
config/
├── __init__.py
├── settings.py              # Main settings (Pydantic)
├── development.py           # Dev environment
├── staging.py               # Staging environment
├── production.py            # Production environment
├── providers.yaml           # Provider registry
├── pipelines.yaml           # Pipeline configurations
└── logging.yaml             # Logging configuration
```

**Purpose**: Centralized configuration using Pydantic Settings for type safety and validation.

**Usage**:
```python
from config.settings import settings

# Access configuration
api_key = settings.openai_api_key
chunk_size = settings.chunk_size
```

---

#### 🔧 src/core/ - Base Abstractions

```
src/core/
├── __init__.py
├── base.py                  # BaseComponent class
├── interfaces.py            # Protocol definitions
├── exceptions.py            # Custom exceptions
├── constants.py             # Application constants
├── types.py                 # Type definitions
└── registry.py              # Component registry
```

**Purpose**: Foundation layer that all other components build upon.

**Key Classes**:
- `BaseComponent`: Base class with lifecycle management
- `CompletionProvider`: Protocol for text generation
- `EmbeddingProvider`: Protocol for embeddings
- `StorageProvider`: Protocol for storage backends

**Example**:
```python
from src.core.base import BaseComponent

class MyComponent(BaseComponent):
    def initialize(self):
        # Setup resources
        pass
    
    def cleanup(self):
        # Cleanup resources
        pass
    
    # Use as context manager
    with MyComponent(config) as component:
        component.do_work()
```

---

#### 🔌 src/providers/ - External Service Integrations

```
src/providers/
├── __init__.py
├── base.py                  # Base provider
├── completion/              # Text generation providers
│   ├── openai.py
│   ├── anthropic.py
│   ├── google.py
│   ├── cohere.py
│   └── local.py
├── embedding/               # Embedding providers
│   ├── openai.py
│   ├── cohere.py
│   ├── huggingface.py
│   └── sentence_transformers.py
└── multimodal/              # Multimodal providers
    ├── vision.py
    └── audio.py
```

**Purpose**: Unified interface to different AI providers.

**Key Features**:
- Provider-agnostic API
- Automatic retry & fallback
- Cost tracking
- Streaming support

**Example**:
```python
from src.providers.completion import OpenAIProvider, AnthropicProvider

# Use any provider with same interface
for provider_class in [OpenAIProvider, AnthropicProvider]:
    provider = provider_class(model="best-model")
    response = provider.complete("Hello, world!")
    print(response)
```

---

#### 📊 src/data/ - Data Operations

```
src/data/
├── sources/                 # Data source connectors
│   ├── files.py            # CSV, JSON, Parquet
│   ├── documents.py        # PDF, DOCX, TXT
│   ├── databases.py        # SQL, NoSQL
│   ├── apis.py             # REST, GraphQL
│   ├── streams.py          # Kafka, Kinesis
│   └── web.py              # Web scraping
├── processors/              # Data processing
│   ├── text.py
│   ├── structured.py
│   ├── media.py
│   └── batch.py
├── transformers/            # Data transformations
│   ├── chunking.py
│   ├── cleaning.py
│   ├── enrichment.py
│   └── aggregation.py
├── validators/              # Data validation
│   ├── schema.py
│   ├── quality.py
│   └── integrity.py
└── serializers/             # Data serialization
    ├── json.py
    ├── avro.py
    └── parquet.py
```

**Purpose**: Handle all data loading, processing, and transformation.

**Example**:
```python
from src.data.sources import DocumentSource
from src.data.processors import TextProcessor
from src.data.transformers import Chunking

# Load documents
source = DocumentSource()
docs = source.load("./documents/*.pdf")

# Process and transform
processor = TextProcessor()
cleaned_docs = processor.clean(docs)

chunker = Chunking(strategy="semantic", chunk_size=512)
chunks = chunker.transform(cleaned_docs)
```

---

#### 💾 src/storage/ - Persistence Layer

```
src/storage/
├── databases/               # Database connections
│   ├── relational.py       # PostgreSQL, MySQL, SQLite
│   ├── document.py         # MongoDB, DynamoDB
│   ├── graph.py            # Neo4j, ArangoDB
│   └── columnar.py         # Cassandra, HBase
├── indexes/                 # Search & indexing
│   ├── vector/             # Vector similarity
│   │   ├── faiss.py
│   │   ├── chroma.py
│   │   ├── pinecone.py
│   │   ├── qdrant.py
│   │   └── weaviate.py
│   └── fulltext/           # Full-text search
│       ├── elasticsearch.py
│       └── opensearch.py
├── cache/                   # Caching layer
│   ├── memory.py
│   ├── redis.py
│   └── disk.py
├── objects/                 # Object/blob storage
│   ├── s3.py
│   ├── gcs.py
│   └── local.py
└── queues/                  # Message queues
    ├── rabbitmq.py
    ├── kafka.py
    └── sqs.py
```

**Purpose**: Unified interface to all storage backends.

**Example**:
```python
from src.storage.indexes.vector import ChromaStore, PineconeStore

# Same interface for different stores
for store_class in [ChromaStore, PineconeStore]:
    store = store_class(collection="docs")
    store.upsert(
        ids=["doc1", "doc2"],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadata
    )
    
    results = store.search(query_embedding, top_k=5)
```

---

#### 🔍 src/retrieval/ - Information Retrieval

```
src/retrieval/
├── strategies/              # Retrieval strategies
│   ├── semantic.py         # Vector search
│   ├── lexical.py          # BM25, TF-IDF
│   ├── hybrid.py           # Combined approaches
│   ├── graph.py            # Graph traversal
│   └── sql.py              # SQL queries
├── ranking/                 # Result ranking
│   ├── cross_encoder.py    # Neural reranking
│   ├── fusion.py           # Result fusion
│   └── relevance.py        # Relevance scoring
├── query/                   # Query processing
│   ├── parser.py
│   ├── expander.py
│   ├── router.py
│   └── optimizer.py
└── filters/                 # Result filtering
    ├── metadata.py
    ├── temporal.py
    └── semantic.py
```

**Purpose**: Retrieve relevant information from storage.

**Example**:
```python
from src.retrieval.strategies import HybridRetrieval
from src.retrieval.ranking import CrossEncoderReranker

# Hybrid search
retriever = HybridRetrieval(
    vector_store=vector_store,
    vector_weight=0.7,
    lexical_weight=0.3
)

results = retriever.search("machine learning", top_k=20)

# Rerank results
reranker = CrossEncoderReranker()
final_results = reranker.rerank(results, top_k=5)
```

---

#### ⚙️ src/execution/ - Task Execution Engine

```
src/execution/
├── engines/                 # Execution engines
│   ├── sequential.py       # Sequential execution
│   ├── parallel.py         # Parallel execution
│   ├── conditional.py      # Conditional branching
│   ├── loop.py             # Iterative execution
│   └── reactive.py         # Event-driven execution
├── tasks/                   # Task definitions
│   ├── compute.py          # Computation tasks
│   ├── io.py               # I/O operations
│   ├── transform.py        # Data transformations
│   └── api.py              # API calls
├── orchestration/           # Workflow orchestration
│   ├── dag.py              # DAG executor
│   ├── state_machine.py    # State machine
│   ├── coordinator.py      # Multi-task coordination
│   └── scheduler.py        # Task scheduling
└── state/                   # State management
    ├── manager.py          # State manager
    ├── context.py          # Execution context
    └── snapshot.py         # State snapshots
```

**Purpose**: Execute tasks with different patterns (sequential, parallel, conditional, etc.).

**Example**:
```python
from src.execution.engines import ParallelEngine, SequentialEngine
from src.execution.tasks import APITask, TransformTask

# Define tasks
fetch_task = APITask(endpoint="https://api.example.com/data")
transform_task = TransformTask(transformer=my_transformer)

# Execute in parallel
parallel_engine = ParallelEngine(tasks=[fetch_task, fetch_task2])
results = parallel_engine.run()

# Then execute sequentially
sequential_engine = SequentialEngine(tasks=[transform_task, save_task])
final_result = sequential_engine.run(results)
```

---

#### 🧠 src/reasoning/ - Decision-Making Logic

```
src/reasoning/
├── strategies/              # Reasoning strategies
│   ├── chain.py            # Chain-of-thought
│   ├── tree.py             # Tree-of-thought
│   ├── react.py            # ReAct pattern
│   ├── plan_execute.py     # Plan & Execute
│   └── reflection.py       # Self-reflection
├── tools/                   # Executable tools
│   ├── base.py
│   ├── search.py           # Web/document search
│   ├── calculation.py      # Math operations
│   ├── database.py         # Database queries
│   ├── api.py              # API interactions
│   ├── file.py             # File operations
│   ├── code.py             # Code execution
│   └── shell.py            # Shell commands
├── planning/                # Task planning
│   ├── decomposition.py    # Task breakdown
│   ├── prioritization.py   # Priority assignment
│   └── optimization.py     # Plan optimization
└── selection/               # Decision selection
    ├── scoring.py          # Option scoring
    ├── ranking.py          # Option ranking
    └── filtering.py        # Option filtering
```

**Purpose**: Implement reasoning patterns for decision-making.

**Example**:
```python
from src.reasoning.strategies import ReactStrategy
from src.reasoning.tools import SearchTool, CalculatorTool, DatabaseTool

# Define available tools
tools = [
    SearchTool(api_key="..."),
    CalculatorTool(),
    DatabaseTool(connection="...")
]

# Create reasoning strategy
strategy = ReactStrategy(
    provider=completion_provider,
    tools=tools,
    max_iterations=10
)

# Execute reasoning
result = strategy.reason("What's 15% of Tokyo's current population?")
print(f"Answer: {result.answer}")
print(f"Steps: {result.steps}")
```

---

#### 🧬 src/context/ - State Management

```
src/context/
├── memory/                  # Memory systems
│   ├── buffer.py           # Recent context buffer
│   ├── sliding.py          # Sliding window
│   ├── summary.py          # Summarized history
│   ├── semantic.py         # Semantic memory
│   ├── entity.py           # Entity tracking
│   └── knowledge.py        # Knowledge base
├── session/                 # Session management
│   ├── manager.py          # Session lifecycle
│   ├── storage.py          # Session persistence
│   └── recovery.py         # Session recovery
└── persistence/             # Context persistence
    ├── checkpoint.py       # Checkpoint system
    ├── snapshot.py         # State snapshots
    └── replay.py           # Event replay
```

**Purpose**: Manage conversational context, session state, and memory.

**Example**:
```python
from src.context.memory import BufferMemory, SemanticMemory

# Short-term memory
buffer = BufferMemory(max_messages=10)
buffer.add_message({"role": "user", "content": "Hello"})
buffer.add_message({"role": "assistant", "content": "Hi!"})

# Long-term semantic memory
semantic_memory = SemanticMemory(vector_store=store)
semantic_memory.store("Important fact: User prefers dark mode")

# Retrieve relevant memories
relevant = semantic_memory.retrieve("What are user preferences?")
```

---

#### 💬 src/interaction/ - User/System Communication

```
src/interaction/
├── conversations/           # Conversation management
│   ├── manager.py          # Conversation lifecycle
│   ├── turns.py            # Turn management
│   └── history.py          # Conversation history
├── protocols/               # Communication protocols
│   ├── mcp/                # Model Context Protocol
│   │   ├── server.py
│   │   ├── client.py
│   │   ├── resources.py
│   │   └── transports/
│   │       ├── stdio.py
│   │       ├── sse.py
│   │       └── websocket.py
│   ├── openai/             # OpenAI-compatible API
│   │   └── compatible.py
│   └── custom/             # Custom protocols
├── streaming/               # Response streaming
│   ├── server_sent_events.py
│   └── websocket.py
└── formatting/              # Response formatting
    ├── text.py
    ├── markdown.py
    ├── html.py
    └── structured.py
```

**Purpose**: Handle all user and system interactions.

**Example**:
```python
from src.interaction.conversations import ConversationManager
from src.context.memory import BufferMemory

manager = ConversationManager(
    provider=completion_provider,
    memory=BufferMemory(max_messages=10)
)

# Send message
response = manager.send_message("What's the weather?")
print(response.content)

# Stream response
for chunk in manager.stream_message("Tell me a story"):
    print(chunk, end="", flush=True)

# Export conversation
manager.export_history("conversation.json")
```

---

#### 🔄 src/pipelines/ - Workflow Orchestration

```
src/pipelines/
├── __init__.py
├── base.py                  # Base pipeline class
├── ingestion.py             # Data ingestion
├── processing.py            # Data processing
├── indexing.py              # Index building
├── inference.py             # Model inference
├── serving.py               # Model serving
└── composite.py             # Composite pipelines
```

**Purpose**: Orchestrate end-to-end workflows.

**Example**:
```python
from src.pipelines import IngestionPipeline, InferencePipeline, CompositePipeline

# Define individual pipelines
ingestion = IngestionPipeline(
    source=document_source,
    transformers=[chunker, embedder],
    destination=vector_store
)

inference = InferencePipeline(
    retriever=retriever,
    provider=completion_provider
)

# Combine into composite pipeline
pipeline = CompositePipeline(stages=[ingestion, inference])

# Execute
result = pipeline.run(query="What is machine learning?")
```

---

#### 📊 src/monitoring/ - Observability

```
src/monitoring/
├── metrics/                 # Metrics collection
│   ├── performance.py      # Latency, throughput
│   ├── resource.py         # CPU, memory, GPU
│   ├── quality.py          # Data/output quality
│   ├── cost.py             # Cost tracking
│   └── business.py         # Business metrics
├── logging/                 # Logging system
│   ├── structured.py       # Structured logging
│   ├── correlation.py      # Request correlation
│   └── aggregation.py      # Log aggregation
├── tracing/                 # Distributed tracing
│   ├── tracer.py           # Tracer implementation
│   ├── spans.py            # Span management
│   ├── propagation.py      # Context propagation
│   └── exporters/          # Trace exporters
│       ├── jaeger.py
│       ├── zipkin.py
│       └── otlp.py
└── alerting/                # Alert management
    ├── rules.py            # Alert rules
    ├── handlers.py         # Alert handlers
    └── channels/           # Notification channels
        ├── email.py
        ├── slack.py
        └── pagerduty.py
```

**Purpose**: Monitor application health, performance, and usage.

**Example**:
```python
from src.monitoring.metrics import MetricsCollector
from src.monitoring.logging import StructuredLogger
from src.monitoring.tracing import Tracer

# Collect metrics
metrics = MetricsCollector()
metrics.record("api_request", 1.0, labels={"endpoint": "/completion"})
metrics.record("latency_ms", 523.4)

# Structured logging
logger = StructuredLogger("app")
logger.info("Request processed", user_id="123", duration=0.523)

# Distributed tracing
tracer = Tracer()
with tracer.start_span("process_request") as span:
    span.set_attribute("user_id", "123")
    # ... processing logic
```

---

#### 🧪 src/evaluation/ - Quality Evaluation

```
src/evaluation/
├── metrics/                 # Evaluation metrics
│   ├── retrieval.py        # Precision, recall, MRR
│   ├── generation.py       # BLEU, ROUGE, coherence
│   ├── task_specific.py    # Custom metrics
│   └── custom.py
├── judges/                  # Automated judges
│   ├── model_based.py      # Model-as-judge
│   ├── rule_based.py       # Rule-based evaluation
│   └── hybrid.py           # Combined approaches
├── benchmarks/              # Benchmark suites
│   ├── standard.py         # Standard benchmarks
│   └── custom.py           # Custom benchmarks
└── experiments/             # A/B testing
    ├── manager.py          # Experiment manager
    ├── variants.py         # Variant management
    └── analysis.py         # Results analysis
```

**Purpose**: Evaluate system quality and performance.

**Example**:
```python
from src.evaluation.metrics import RetrievalMetrics
from src.evaluation.judges import ModelBasedJudge

# Evaluate retrieval quality
retrieval_metrics = RetrievalMetrics()
precision = retrieval_metrics.precision(retrieved, relevant)
recall = retrieval_metrics.recall(retrieved, relevant)
mrr = retrieval_metrics.mean_reciprocal_rank(results, relevant)

# Evaluate generation quality
judge = ModelBasedJudge(provider=gpt4_provider)
faithfulness = judge.evaluate_faithfulness(
    answer=generated_answer,
    context=retrieved_context
)
```

---

#### 🔒 src/security/ - Security & Compliance

```
src/security/
├── authentication/          # Authentication
│   ├── jwt.py              # JWT auth
│   ├── oauth.py            # OAuth2
│   ├── api_key.py          # API key auth
│   └── mutual_tls.py       # mTLS
├── authorization/           # Authorization
│   ├── rbac.py             # Role-based access
│   ├── abac.py             # Attribute-based access
│   └── policies.py         # Policy engine
├── validation/              # Input/output validation
│   ├── input_sanitizer.py  # Input sanitization
│   ├── output_filter.py    # Output filtering
│   ├── injection_detector.py  # Injection detection
│   └── content_moderation.py  # Content moderation
├── encryption/              # Encryption utilities
│   ├── symmetric.py        # Symmetric encryption
│   ├── asymmetric.py       # Asymmetric encryption
│   └── hashing.py          # Hashing functions
└── secrets/                 # Secret management
    ├── manager.py          # Secret manager
    ├── vault.py            # Vault integration
    └── rotation.py         # Secret rotation
```

**Purpose**: Ensure security and compliance.

**Example**:
```python
from src.security.authentication import JWTAuthenticator
from src.security.validation import InputSanitizer

# Authentication
auth = JWTAuthenticator(secret_key=settings.secret_key)
token = auth.create_token({"user_id": "123"})
payload = auth.verify_token(token)

# Input validation
sanitizer = InputSanitizer()
is_valid, error, clean_input = sanitizer.validate_input(
    user_input,
    max_length=2000,
    check_injections=True
)
```

---

#### 🛠️ src/utilities/ - Helper Utilities

```
src/utilities/
├── text/                    # Text utilities
│   ├── tokenization.py
│   ├── normalization.py
│   ├── extraction.py
│   └── similarity.py
├── network/                 # Network utilities
│   ├── http_client.py
│   ├── retry.py
│   ├── rate_limit.py
│   ├── circuit_breaker.py
│   └── timeout.py
├── serialization/           # Serialization
│   ├── json.py
│   ├── yaml.py
│   ├── pickle.py
│   └── msgpack.py
├── concurrency/             # Concurrency utilities
│   ├── locks.py
│   ├── pools.py
│   └── async_helpers.py
└── helpers/                 # General helpers
    ├── datetime.py
    ├── filesystem.py
    ├── hashing.py
    └── validation.py
```

**Purpose**: Common utilities used across the platform.

**Example**:
```python
from src.utilities.text import Tokenizer
from src.utilities.network import HTTPClient, RetryHandler

# Tokenization
tokenizer = Tokenizer(model="gpt-4")
token_count = tokenizer.count_tokens("Hello, world!")

# HTTP with retry
client = HTTPClient(retry_handler=RetryHandler(max_attempts=3))
response = client.get("https://api.example.com/data")
```

---

## 🔄 Data Flow Patterns

### Pattern 1: Simple Completion

```
User Input
    ↓
Provider (OpenAI/Anthropic)
    ↓
Response
```

**Code**:
```python
from src.providers.completion import OpenAIProvider

provider = OpenAIProvider(model="gpt-4")
response = provider.complete("Hello!")
```

---

### Pattern 2: RAG (Retrieval Augmented Generation)

```
User Query
    ↓
Embedding Provider
    ↓
Vector Store Search
    ↓
Retrieved Documents
    ↓
Completion Provider (with context)
    ↓
Response
```

**Code**:
```python
# Index documents
docs = DocumentSource().load("./docs/*.pdf")
embedder = OpenAIEmbedding()
store = ChromaStore()
store.index(docs, embedder)

# Query
query = "What is machine learning?"
query_embedding = embedder.embed(query)
retrieved = store.search(query_embedding, top_k=5)

# Generate with context
provider = OpenAIProvider()
context = "\n".join([doc.content for doc in retrieved])
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
response = provider.complete(prompt)
```

---

### Pattern 3: Agentic Reasoning (ReAct)

```
User Query
    ↓
Reasoning Strategy
    ↓
┌─────────────────────┐
│ Loop (max N times) │
│                     │
│ 1. Think           │
│ 2. Select Tool     │
│ 3. Execute Tool    │
│ 4. Observe Result  │
└─────────────────────┘
    ↓
Final Answer
```

**Code**:
```python
from src.reasoning.strategies import ReactStrategy
from src.reasoning.tools import SearchTool, CalculatorTool

tools = [SearchTool(), CalculatorTool()]
strategy = ReactStrategy(provider=provider, tools=tools)

result = strategy.reason("What's 15% of Tokyo's population?")
# Agent will:
# 1. Search for Tokyo population
# 2. Calculate 15% of that number
# 3. Return answer
```

---

### Pattern 4: Multi-Stage Pipeline

```
Data Source
    ↓
Extract
    ↓
Transform
    ↓
Load to Vector Store
    ↓
Query
    ↓
Retrieve
    ↓
Rerank
    ↓
Generate
    ↓
Response
```

**Code**:
```python
from src.pipelines import CompositePipeline

pipeline = CompositePipeline(stages=[
    IngestionPipeline(source=source, destination=store),
    RetrievalPipeline(retriever=retriever, reranker=reranker),
    InferencePipeline(provider=provider)
])

result = pipeline.run(query="Your question")
```

---

## 🔧 Extension Guide

### Adding a New Provider

**1. Create provider class**:

```python
# src/providers/completion/custom.py
from src.core.base import BaseComponent
from src.core.interfaces import CompletionProvider

class CustomProvider(BaseComponent, CompletionProvider):
    """Custom completion provider"""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.client = None
    
    def initialize(self):
        """Initialize the provider"""
        self.client = CustomClient(api_key=self.api_key)
        self._initialized = True
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion"""
        self.validate_initialized()
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            **kwargs
        )
        return response.text
    
    def stream(self, prompt: str, **kwargs):
        """Stream completion"""
        self.validate_initialized()
        for chunk in self.client.stream_generate(
            model=self.model,
            prompt=prompt,
            **kwargs
        ):
            yield chunk.text
    
    def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()
```

**2. Register provider**:

```python
from src.core.registry import ComponentRegistry
from src.providers.completion.custom import CustomProvider

ComponentRegistry.register("custom", CustomProvider)
```

**3. Use provider**:

```python
from src.core.registry import ComponentRegistry

provider = ComponentRegistry.create("custom", {
    "api_key": "your-key",
    "model": "custom-model"
})
response = provider.complete("Hello!")
```

---

### Adding a New Tool

**1. Create tool class**:

```python
# src/reasoning/tools/weather.py
from src.reasoning.tools.base import BaseTool

class WeatherTool(BaseTool):
    """Tool to get weather information"""
    
    name = "weather"
    description = "Get current weather for a location"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def execute(self, location: str) -> str:
        """Get weather for location"""
        # Call weather API
        weather_data = self._call_weather_api(location)
        return f"Weather in {location}: {weather_data['condition']}, {weather_data['temp']}°C"
    
    def _call_weather_api(self, location: str):
        # API call logic
        pass
```

**2. Use tool in reasoning**:

```python
from src.reasoning.strategies import ReactStrategy
from src.reasoning.tools.weather import WeatherTool

tools = [WeatherTool(api_key="your-key")]
strategy = ReactStrategy(provider=provider, tools=tools)

result = strategy.reason("What's the weather in Tokyo?")
```

---

### Adding a New Vector Store

**1. Create store class**:

```python
# src/storage/indexes/vector/custom_store.py
from src.storage.indexes.vector.base import BaseVectorStore

class CustomVectorStore(BaseVectorStore):
    """Custom vector store implementation"""
    
    def __init__(self, collection: str, **kwargs):
        super().__init__(**kwargs)
        self.collection = collection
        self.client = None
    
    def initialize(self):
        """Initialize connection"""
        self.client = CustomVectorClient()
        self._initialized = True
    
    def upsert(self, ids, embeddings, documents, metadatas=None):
        """Insert or update vectors"""
        self.validate_initialized()
        self.client.upsert(
            collection=self.collection,
            ids=ids,
            vectors=embeddings,
            payloads=metadatas
        )
    
    def search(self, query_embedding, top_k=5, filters=None):
        """Search for similar vectors"""
        self.validate_initialized()
        results = self.client.search(
            collection=self.collection,
            vector=query_embedding,
            limit=top_k,
            filter=filters
        )
        return self._format_results(results)
    
    def delete(self, ids):
        """Delete vectors"""
        self.validate_initialized()
        self.client.delete(collection=self.collection, ids=ids)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()
```

**2. Use the store**:

```python
from src.storage.indexes.vector.custom_store import CustomVectorStore

store = CustomVectorStore(collection="my_docs")
store.upsert(ids=["1", "2"], embeddings=embeddings, documents=texts)
results = store.search(query_embedding, top_k=5)
```

---

## 📋 Best Practices

### 1. Configuration Management

✅ **Do**:
```python
# Use centralized configuration
from config.settings import settings

api_key = settings.openai_api_key
chunk_size = settings.chunk_size
```

❌ **Don't**:
```python
# Don't hardcode values
api_key = "sk-..."  # Hard-coded key
chunk_size = 512    # Magic number
```

### 2. Error Handling

✅ **Do**:
```python
from src.core.exceptions import ProviderError

try:
    response = provider.complete(prompt)
except ProviderError as e:
    logger.error("Provider failed", error=str(e))
    # Fallback logic
    response = fallback_provider.complete(prompt)
```

❌ **Don't**:
```python
try:
    response = provider.complete(prompt)
except Exception:
    pass  # Silent failure
```

### 3. Resource Management

✅ **Do**:
```python
# Use context managers
with provider:
    response = provider.complete(prompt)
# Automatic cleanup
```

❌ **Don't**:
```python
provider = Provider()
response = provider.complete(prompt)
# Forget to cleanup
```

### 4. Logging

✅ **Do**:
```python
from src.monitoring.logging import StructuredLogger

logger = StructuredLogger("app")
logger.info("Request processed", 
    user_id="123",
    duration=0.523,
    tokens=1500
)
```

❌ **Don't**:
```python
print(f"Request processed for user 123")  # Unstructured
```

### 5. Testing

✅ **Do**:
```python
# Write unit tests for components
def test_provider_completion():
    provider = MockProvider()
    response = provider.complete("test")
    assert response == "expected"

# Write integration tests for workflows
def test_rag_pipeline():
    pipeline = RAGPipeline()
    result = pipeline.run(query="test")
    assert result.answer is not None
```

### 6. Type Hints

✅ **Do**:
```python
from typing import List, Optional

def process_documents(
    docs: List[str],
    chunk_size: int = 512,
    overlap: Optional[int] = None
) -> List[str]:
    # Implementation
    pass
```

❌ **Don't**:
```python
def process_documents(docs, chunk_size=512, overlap=None):
    # No type hints
    pass
```

---

## 📚 Additional Resources

- **API Reference**: See `docs/api-reference/` for detailed API documentation
- **Examples**: See `examples/` for working code samples
- **Tutorials**: See `docs/tutorials/` for step-by-step guides
- **Contributing**: See `CONTRIBUTING.md` for contribution guidelines

---

## 🔗 Quick Links

- [README](README.md) - Project overview
- [Quick Start](docs/getting-started/quickstart.md) - Get started quickly
- [Configuration Guide](docs/getting-started/configuration.md) - Configure the platform
- [Examples](examples/) - Code examples
- [API Reference](docs/api-reference/) - Complete API docs

---