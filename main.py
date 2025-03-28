import argparse
import json
import os

def main():
    
    
    parser = argparse.ArgumentParser(description="TinyRAG Proof of Concept")
    parser.add_argument("--query", type=str, help="Query for the AI RAG system")
    args = parser.parse_args()

    if args.query:
        response = process_query(args.query)
        print("\nAI Response:\n", response)
    else:
        print("No query provided. Run with --query 'your question'.")

def process_query(query):
    """
    Placeholder function for processing a query.
    Replace with actual FAISS retrieval + LLM inference logic.
    """
    return f"Simulated response for: {query}"

def load_wikipedia_data(json_dir):
    docs = []
    for file in os.listdir(json_dir):
        file_path = os.path.join(json_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                title = data.get("title", "")
                text = data.get("text", "")
                if title and text:
                    docs.append({"title": title, "text": text})
    return docs

if __name__ == "__main__":
    main()
