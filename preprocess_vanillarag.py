import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Define directories
DATA_DIR = "./wiki_data"
INDEX_PATH = "./faiss_index"
DOCS_PATH = "./documents.json"

# Initialize embedding model (small and efficient)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Storage for documents
documents = []
embeddings = []

def process_file(file_path):
    """Extract text from JSON and compute embeddings."""
    try:
        with open(file_path, "r", encoding="utf-8") as json_file:
            for line in json_file:
                data = json.loads(line)
                title = data.get("title", "").strip()
                text = data.get("text", "").strip()
                
                if title and text:
                    full_text = f"{title}\n\n{text}"
                    documents.append(full_text)

                    # Compute embedding
                    embedding = embedder.encode(full_text, convert_to_numpy=True)
                    embeddings.append(embedding)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def traverse_and_process(data_dir):
    """Traverse directories and process all files."""
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)

        if os.path.isdir(subfolder_path):
            for file in tqdm(os.listdir(subfolder_path), desc="Processing files"):
                file_path = os.path.join(subfolder_path, file)
                if os.path.isfile(file_path) and "." not in file:  # Ensure it's a data file
                    process_file(file_path)

def build_faiss_index():
    """Create and save a FAISS index."""
    if not embeddings:
        print("No embeddings to index.")
        return
    
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    
    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)

    # Save documents
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f)

    print(f"Indexed {len(documents)} documents.")

# Run preprocessing
