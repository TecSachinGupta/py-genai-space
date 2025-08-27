# Repository Structure Extension Guide

This guide explains how to expand your data engineering repository structure based on your project's evolving needs.

## Core Philosophy

- **Start minimal**: Only create folders/files you actively use
- **Expand purposefully**: Add structure when you have 3+ related files
- **Document changes**: Update this guide when you customize the structure

## Quick Expansion Commands

### Add GenAI capabilities
```bash
mkdir -p src/utilities/genai/{llm_clients,prompts,embeddings,retrieval}
mkdir -p src/jobs/genai/{rag_pipelines,content_generation}
```

### Add ML capabilities  
```bash
mkdir -p src/utilities/ml
mkdir -p src/jobs/ml/{training,inference}
mkdir -p src/notebooks/02b_genai_modeling
```

### Add advanced testing
```bash
mkdir -p tests/{e2e,performance}
```

For detailed expansion guidelines, see the full STRUCTURE_GUIDE.md documentation.
