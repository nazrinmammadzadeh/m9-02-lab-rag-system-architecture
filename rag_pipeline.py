import json
import os
from google import genai
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv


# Setup and configuration
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

client = genai.Client(api_key=api_key)


chroma_client = chromadb.Client()

embedding_func = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="kb_collection",
    embedding_function=embedding_func
)



# Indexing
def index_knowledge_base(file_path):
    with open(file_path, 'r') as f:
        kb_data = json.load(f)

    documents = []
    metadatas = []
    ids = []

    '''
    We don't need to split the text today because each JSON entry is already a short, perfect chunk.
    If these were full, long documents (like books or PDFs), we would have to chunk them because:
    1. Big documents break the token size limits of embedding models and LLMs.
    2. Embedding huge files blends all concepts together, making search inaccurate.
    3. Small chunks keep search results precise, targeted, and cheaper to process.
    '''

    for entry in kb_data:
        content = entry.get("text")
        source = entry.get("source", "Unknown Source")
        doc_id = entry.get("id")

        if content and doc_id:
            documents.append(content)
            metadatas.append({"source": source})
            ids.append(doc_id)

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

# Query and generation
def run_rag_pipeline(question, n_results=3):
    print(f"User question: '{question}'")

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    retrieved_docs = results['documents'][0]
    retrieved_meta = results['metadatas'][0]

    print("\nRetrieved Sources")
    context_str = ""
    for i, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_meta), 1):
        source_name = meta.get('source', 'Unknown')
        print(f"[{i}] Source: {source_name}")
        print(f"    Text snippet: {doc[:120]}...")

        context_str += f"Source: {source_name}\nContent: {doc}\n\n"

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided text context below. 

CRITICAL RULES:
1. Every factual claim you make must cite its source using the exact format: [Source Name].
2. If the answer cannot be completely and truthfully found in the context below, you must reply exactly with: "I don't know." Do not try to make up or infer an answer from outside knowledge.

Context:
{context_str}

Question: {question}
Answer:"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    print("\nFinal Answer")
    print(response.text)
    print("-------------------------------------------------\n")

# Running the evaluation
if __name__ == "__main__":
    index_knowledge_base("knowledge_base.json")

    questions = [
        "How long do I have to get a full refund?",
        "How do I reset my password?",
        "What is the company's stock price today?"
    ]

    for q in questions:
        run_rag_pipeline(q, n_results=3)