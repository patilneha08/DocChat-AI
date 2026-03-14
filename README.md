# 📄 DocChat AI - RAG based document AI assist

A **RAG-powered document AI assist** that lets you upload documents and ask questions 
grounded in their content. Built with LangChain, Ollama, ChromaDB, and Streamlit.

## Why this project is useful
This system helps users interact with large documents through natural language instead of manually searching through pages of content. By using Retrieval-Augmented Generation (RAG), it retrieves relevant context from uploaded documents and generates accurate, context-aware answers.

## Where this system helps

- Research and study: quickly extract answers from notes, PDFs, and reference materials
- Business documents: query reports, manuals, contracts, and internal knowledge bases
- Customer support/internal ops: retrieve information from policy docs or FAQs
- Productivity: save time by avoiding manual document scanning

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

## Steps to run

```bash
# 1. Clone and set up
git clone https://github.com/patilneha08/DocChat-AI
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
