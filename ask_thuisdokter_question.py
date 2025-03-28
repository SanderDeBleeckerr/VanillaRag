import argparse
import faiss
import json

import numpy as np
from sentence_transformers import SentenceTransformer

# Directory for FAISS index and documents
INDEX_PATH = "./faiss_index_thuisdokter"
DOCS_PATH = "./thuisdokter_documents.json"

# Initialize Sentence-Transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents and embeddings
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)


# Load FAISS index
index = faiss.read_index(INDEX_PATH)

def query_faiss_index(query, k=3):
    """Query FAISS index and retrieve top k most similar documents."""
    # Encode the query to get the query embedding
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # Perform the search to find the most similar documents
    distances, indices = index.search(query_embedding, k)

    # Fetch the documents corresponding to the indices and return the results
    results = [(documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Query the Thuisdokter document system.")
    parser.add_argument("--query", type=str, help="The question you want to ask.", required=True)
    args = parser.parse_args()

    # Run query
    print(f"Querying FAISS index with: '{args.query}'")
    results = query_faiss_index(args.query)

    # Display the results of the query
    print("\nTop matching documents:")
    for i, (doc, score) in enumerate(results):
        print(f"\nDocument {i + 1} (Score: {score}):\n{doc[:500]}...")  # Print first 500 chars of the document

if __name__ == "__main__":
    main()
