# rag-pipeline-dataset

Build a RAG pipeline using dataset containing news information with local LLM integration.

## Project Status

**🚀 Ready for Development**: Core infrastructure is complete, notebook ready for RAG pipeline implementation.

**✅ Completed Components:**

- Local LLM integration with LM Studio
- Utilities for embeddings and retrieval
- Testing framework without external dependencies
- Environment setup and documentation

**🔨 Ready to Build:**

- Main RAG pipeline notebook (currently empty)
- Core pipeline functions (guided exercises available)
- Interactive comparison widgets

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that:

- Uses local embeddings for semantic search through news articles
- Integrates with LM Studio for local LLM inference
- Provides interactive widgets for comparing RAG vs non-RAG responses
- Includes testing utilities for validation

## Project Structure

```
rag-pipeline-dataset/
├── README.md                    # This file
├── .gitignore                  # Git ignore patterns
├── rag_pipeline_news.ipynb     # Main Jupyter notebook
├── utils.py                    # Core utilities and functions
├── unittests.py                # Testing framework
├── news_data_dedup.csv         # News dataset (when available)
├── embeddings.joblib           # Pre-computed embeddings (when available)
└── rag_env/                    # Virtual environment (ignored by git)
```

## Environment Setup

### Creating Virtual Environment

```bash
python3 -m venv rag_env
```

### Activating Virtual Environment

```bash
source rag_env/Scripts/activate
```

### Installing Dependencies

```bash
pip install sentence-transformers pandas numpy scikit-learn joblib requests ipywidgets jupyter
```

### Benefits

Using a virtual environment isolates your project dependencies from your system Python, preventing version conflicts and ensuring your project runs consistently across different machines.

## LM Studio Integration

This project is configured to work with **LM Studio** for local LLM inference.

### Model Setup

- **Model**: `hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF`
- **File**: `llama-3.2-1b-instruct-q8_0.gguf`
- **API Identifier**: `llama-3.2-1b-instruct`
- **Local Server**: `http://127.0.0.1:1234`

### Prerequisites

1. Install and run LM Studio
2. Load the Llama-3.2-1B-Instruct model
3. Start the local server on port 1234
4. Ensure the server is accessible at `http://127.0.0.1:1234`

## Key Components

### `utils.py`

Core utilities including:

- `generate_with_single_input()` - Interface to local LLM via LM Studio
- `retrieve()` - Semantic search using embeddings
- `read_dataframe()` - News data processing
- `display_widget()` - Interactive comparison interface
- `concatenate_fields()` - Text preprocessing

### `unittests.py`

Testing framework with:

- `test_format_relevant_data()` - Validates data formatting
- `test_get_relevant_data()` - Tests retrieval functionality
- Mock grading system (no external dependencies)

### `rag_pipeline_news.ipynb`

**Status**: Currently empty - ready for development

Intended for:

- RAG pipeline demonstration and exercises
- Step-by-step implementation of core functions
- Interactive testing and comparison of RAG vs non-RAG responses
- End-to-end workflow development

## Usage

### Getting Started

1. **Activate environment**:

   ```bash
   source rag_env/Scripts/activate
   ```

2. **Start LM Studio** with the configured model

3. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

4. **Open** `rag_pipeline_news.ipynb` (currently empty)

### Building the RAG Pipeline

The notebook is ready for development. To build a complete RAG pipeline, you'll need to implement:

1. **Import Setup**:

   ```python
   from utils import (
       retrieve, pprint, generate_with_single_input,
       read_dataframe, display_widget, NEWS_DATA
   )
   import unittests
   ```

2. **Core Functions**:

   - `query_news(indices)` - Retrieve documents by indices
   - `get_relevant_data(query, top_k)` - Combine retrieve + query_news
   - `format_relevant_data(documents)` - Format documents for prompts

3. **Embedding Generation**:

   - Generate embeddings from news data using sentence transformers
   - Save to `embeddings.joblib` for retrieval functionality

4. **RAG Pipeline Testing**:
   - Test retrieval functionality
   - Compare RAG vs non-RAG responses
   - Interactive widgets for experimentation

## Features

- ✅ **Local LLM Integration** - No external API keys required
- ✅ **Semantic Search** - Using sentence transformers for embeddings
- ✅ **Interactive Widgets** - Compare RAG vs non-RAG responses
- ✅ **Robust Error Handling** - Graceful fallbacks for missing files
- ✅ **Testing Framework** - Built-in validation utilities
- ✅ **Clean Dependencies** - No external grading dependencies

## Troubleshooting

### Common Issues

1. **Model Path Error**:

   - The code automatically handles missing `MODEL_PATH` environment variable
   - Uses default model name if not set

2. **Missing Files**:

   - `embeddings.joblib` - Will show warning, generate embeddings first
   - `news_data_dedup.csv` - Provide your news dataset

3. **LM Studio Connection**:

   - Ensure LM Studio is running on `http://127.0.0.1:1234`
   - Check that the model is loaded and server is started

4. **Import Errors**:
   - All external dependencies have been removed or mocked
   - Install required packages in your virtual environment

## Next Steps

### Immediate Setup

1. **Prepare your news dataset** as `news_data_dedup.csv`
2. **Create notebook content** - The `rag_pipeline_news.ipynb` is currently empty and ready for development
3. **Implement core functions** following the building guide above

### Development Workflow

1. **Start with imports** and basic setup
2. **Implement `query_news` function** for document retrieval by indices
3. **Generate embeddings** from your news data using the sentence transformer
4. **Test the `retrieve` function** with sample queries
5. **Implement `get_relevant_data`** combining retrieve + query_news
6. **Create `format_relevant_data`** for prompt formatting
7. **Build complete RAG pipeline** with LM Studio integration
8. **Add interactive widgets** for comparing RAG vs non-RAG responses

### Testing and Validation

- Use the `unittests.py` functions to validate implementations
- Test with various queries and top_k values
- Compare RAG-enhanced vs standard LLM responses
- Customize prompts and parameters for your specific use case

### Ready-to-Use Components

- ✅ `utils.py` - Complete with LM Studio integration
- ✅ `unittests.py` - Testing framework ready
- ✅ Virtual environment setup guide
- ✅ Dependencies and configuration documented
