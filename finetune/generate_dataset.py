"""
Generate a Q&A fine-tuning dataset from your documents using Ollama.
Reads chunks from ChromaDB and asks the LLM to generate question-answer pairs.
Output: finetune/dataset.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_ollama import ChatOllama
import config
from ingest import load_vectorstore

OUTPUT_PATH = Path(__file__).parent / "dataset.jsonl"
NUM_QA_PER_CHUNK = 2


def generate_qa_pairs(chunk_text: str, llm: ChatOllama) -> list[dict]:
    """Ask the LLM to produce Q&A pairs from a chunk."""
    prompt = f"""Based on the following text, generate exactly {NUM_QA_PER_CHUNK} 
question-answer pairs. Return ONLY valid JSON — an array of objects with 
"question" and "answer" keys. No markdown, no explanation.

Text:
{chunk_text}

JSON:"""

    response = llm.invoke(prompt)
    text = response.content.strip()

    # Clean potential markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

    try:
        pairs = json.loads(text)
        if isinstance(pairs, list):
            return [p for p in pairs if "question" in p and "answer" in p]
    except json.JSONDecodeError:
        print(f"  [warn] Could not parse LLM output, skipping chunk.")
    return []


def main(collection: str = "default"):
    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.7,
    )

    vectorstore = load_vectorstore(collection=collection)
    all_docs = vectorstore.get()
    documents = all_docs.get("documents", [])

    print(f"[dataset] Found {len(documents)} chunks in collection '{collection}'")

    dataset = []
    for i, chunk in enumerate(documents):
        print(f"  Generating Q&A for chunk {i+1}/{len(documents)}...")
        pairs = generate_qa_pairs(chunk, llm)
        for pair in pairs:
            dataset.append({
                "instruction": pair["question"],
                "input": chunk[:500],       # context window
                "output": pair["answer"],
            })

    # Write JSONL
    with open(OUTPUT_PATH, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"[dataset] Saved {len(dataset)} examples → {OUTPUT_PATH}")


if __name__ == "__main__":
    collection = sys.argv[1] if len(sys.argv) > 1 else "default"
    main(collection)
