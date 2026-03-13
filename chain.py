"""
RAG chain construction.
Combines the retriever with an Ollama LLM to answer questions
grounded in the uploaded document.
"""
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

import config
from ingest import load_vectorstore


# ── Prompt template ──────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a document Q&A assistant. Follow these rules strictly:

1. ONLY use information from the Context below to answer.
2. ALWAYS reply in English.
3. Quote or reference specific parts of the context to support your answer.
4. If the context does not contain the answer, say exactly: "I couldn't find that in the document."
5. Do NOT use any outside knowledge. Do NOT make up information.
6. Keep your answer concise and directly relevant to the question.

Context:
---
{context}
---

Question: {question}

Answer (in English, based only on the context above):"""
)


def get_llm() -> ChatOllama:
    """Instantiate the Ollama-backed LLM."""
    return ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
        top_p=config.LLM_TOP_P,
    )


def build_chain(collection: str = "default") -> ConversationalRetrievalChain:
    """
    Build a conversational RAG chain.
    - Retriever pulls top-k chunks from ChromaDB.
    - Memory keeps a sliding window of chat history.
    """
    vectorstore = load_vectorstore(collection=collection)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVER_K,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
        output_key="answer",
    )

    llm = get_llm()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain