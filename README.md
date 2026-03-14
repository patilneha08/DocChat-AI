# 📄 DocChat AI — Document Q&A Chatbot

A **RAG-powered chatbot** that lets you upload documents and ask questions 
grounded in their content. Built with LangChain, Ollama, ChromaDB, and Streamlit.

## Tech Stack

| Component     | Technology                              |
|---------------|----------------------------------------|
| LLM           | Mistral 7B via Ollama (local, private) |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store  | ChromaDB                                |
| Framework     | LangChain                               |
| UI            | Streamlit                               |
| Fine-tuning   | QLoRA with PEFT + TRL                  |

## Architecture

```
User uploads PDF/TXT/DOCX
        │
        ▼
   ┌──────────┐    split     ┌──────────┐   embed    ┌──────────┐
   │ ingest.py │ ──────────► │  Chunks  │ ────────► │ ChromaDB │
   └──────────┘              └──────────┘            └──────────┘
                                                          │
User asks question                                   retrieve
        │                                                 │
        ▼                                                 ▼
   ┌──────────┐   prompt + context    ┌──────────────────────┐
   │ chain.py │ ────────────────────► │ Ollama (Mistral 7B)  │
   └──────────┘                       └──────────────────────┘
        │                                      │
        ▼                                      ▼
   ┌──────────┐                          Grounded answer
   │  app.py  │  ◄─────────────────────  + source chunks
   └──────────┘
```

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/docchat-ai.git
cd docchat-ai
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Install & start Ollama
# Download from https://ollama.com
ollama pull mistral

# 3. Run the chatbot
streamlit run app.py
```

## Fine-Tuning (Optional)

```bash
# Install fine-tuning dependencies
pip install transformers datasets peft trl bitsandbytes accelerate torch

# Generate training data from your ingested documents
python finetune/generate_dataset.py <collection_name>

# Run QLoRA fine-tuning
python finetune/finetune.py
```

## Project Structure

```
docchat-ai/
├── app.py                  # Streamlit chat interface
├── ingest.py               # Document loading & vectorization
├── chain.py                # RAG chain with conversational memory
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── README.md
└── finetune/
    ├── generate_dataset.py # Auto-generate Q&A training data
    └── finetune.py         # QLoRA fine-tuning script
```

## License

MIT
