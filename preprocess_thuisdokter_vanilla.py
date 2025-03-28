import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Directory for documents
DOCS_DIR = "./documents"
INDEX_PATH = "./faiss_index_thuisdokter"
# DOCS_PATH = "./thuisdokter_documents.json"

# Initialize Sentence-Transformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Storage for documents and their embeddings
documents = []
embeddings = []

def process_documents():
    """Process the documents in the /documents directory."""
    # Traverse the document directory and process each file
    for filename in tqdm(os.listdir(DOCS_DIR), desc="Processing documents"):
        file_path = os.path.join(DOCS_DIR, filename)

        # Read the file content
        with open(file_path, "r", encoding="utf-8") as file:
            title = filename.replace(".txt", "")  # Use the filename as title
            content = file.read().strip()  # Read content from the file

            if content:
                # Add the document title and content to the list
                full_text = f"{title}\n\n{content}"
                documents.append(full_text)

                # Compute the embedding for the full text of the document
                embedding = embedder.encode(full_text, convert_to_numpy=True)
                embeddings.append(embedding)

def build_faiss_index():
    """Create and save a FAISS index."""
    if not embeddings:
        print("No embeddings to index.")
        return

    dim = len(embeddings[0])  # Get the dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # Create FAISS index (L2 distance)
    index.add(np.array(embeddings, dtype=np.float32))  # Add embeddings to the index

    # Save the FAISS index to disk
    faiss.write_index(index, INDEX_PATH)

    # Save the documents to a JSON file
    # with open(DOCS_PATH, "w", encoding="utf-8") as f:
    #     json.dump(documents, f)

    print(f"Indexed {len(documents)} documents.")

def query_faiss_index(query, k=3):
    """Query FAISS index and retrieve top k most similar documents."""
    # Encode the query to get the query embedding
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # Load the FAISS index from disk
    index = faiss.read_index(INDEX_PATH)

    # Perform the search to find the most similar documents
    distances, indices = index.search(query_embedding, k)

    # Fetch the documents corresponding to the indices and return the results
    results = [(documents[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    
    return results

def main():
    # Step 1: Process all documents and build the FAISS index
    print("Processing documents and building FAISS index...")
    process_documents()
    build_faiss_index()

    # Step 2: Query the FAISS index with a test question
    test_query = "Wat moet ik doen als ik 41 weken zwanger ben?"
    print(f"\nQuerying FAISS index with: '{test_query}'")
    results = query_faiss_index(test_query)

    # Display the results of the query
    print("\nTop matching documents:")
    for i, (doc, score) in enumerate(results):
        print(f"\nDocument {i + 1} (Score: {score}):\n{doc[:500]}...")  # Print first 500 chars of the document

if __name__ == "__main__":
    main()
