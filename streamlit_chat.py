import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Directory for FAISS index and documents
INDEX_PATH = "./faiss_index_thuisdokter"
DOCS_PATH = "./thuisdokter_documents.json"

# Initialize Sentence-Transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents (assuming you stored documents in a JSON file)
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
    """Streamlit app main function."""
    st.title("Thuisdokter Chatbot")
    st.write("Ask me any medical question and I will find the most relevant information.")

    # User input (query)
    user_input = st.text_input("Your Question:", "")

    if user_input:
        # Query the FAISS index for relevant documents
        st.write(f"Searching for: {user_input}")
        results = query_faiss_index(user_input)

        st.subheader("Top matching documents:")

        for i, (doc, score) in enumerate(results):
            st.write(f"**Document {i + 1} (Score: {score})**")
            st.write(f"{doc[:500]}...")  # Show first 500 characters of the document
            st.write("---")

    # Allow user to ask more questions or end the session
    if st.button('Exit'):
        st.write("Goodbye!")

if __name__ == "__main__":
    main()
