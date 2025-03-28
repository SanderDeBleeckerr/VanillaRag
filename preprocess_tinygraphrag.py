import sys
import multiprocessing
import os
import cupy as cp
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This disables GPU usage
import json
import tempfile
import io
# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import ijson
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from tiny_graphrag import store_document, QueryEngine, init_db

# Define the root directory
root_dir = "./wiki_data"
graphs = []

def process_file(file_path):
    """Extract title and text from a JSON file and build the graph."""
    try:
        engine = init_db("postgresql://admin:admin@localhost:5432/tiny-graphrag")

        with open(file_path, "r", encoding="utf-8") as json_file:
            # Stream title and text fields from JSON
            title = None
            text = None

            # Use ijson to efficiently parse fields
            parser = ijson.items(json_file, "", multiple_values=True)
            
            for obj in parser:
                title = obj.get("title", "")
                text = obj.get("text", "")
                if title and text:
                    # Create a temporary file to store the content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
                        temp_file.write(f"{title}\n\n{text}")
                        temp_filepath = temp_file.name

                    try:
                        # Use the in-memory file with the library
                        doc_id, graph_path = store_document(
                            filepath=temp_filepath,
                            title=title,
                            engine=engine
                        )

                        # Load the graph from the graph path
                        graph = nx.read_gpickle(graph_path)

                        # Add the graph to the list
                        graphs.append(graph)
                    finally:
                        # Close the in-memory file (cleanup)
                        os.remove(temp_filepath)
                        
        return f"Processed {file_path}"
    except Exception as e:
        return f"Error processing {file_path}: {e}"

def traverse_and_process(root_dir):
    """Traverse directories and process files in parallel."""
    tasks = []
    processed_count = 0  # Counter to track the number of files processed
    max_files = 4
    
    with ProcessPoolExecutor() as executor:
        # Traverse the directory structure
        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)

            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)

                    # Check for files without extensions
                    if os.path.isfile(file_path) and "." not in file:
                        # Stop if we've reached the maximum number of files
                        if processed_count >= max_files:
                            break
                        
                        # Submit a task for each file
                        tasks.append(executor.submit(process_file, file_path))
                        processed_count += 1

                # Break outer loop if maximum files reached
                if processed_count >= max_files:
                    break
        # Collect results
        for future in tasks:
            print(future.result)
    
    if(graphs):
        # Combine all individual graphs into one large graph
        combined_graph = nx.compose_all(graphs)

        # Save the combined graph for future use
        nx.write_gpickle(combined_graph, "combined_graph.pkl")
    else:
        print("No graphs were processed.")

# Start processing
if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)  # Set the 'spawn' start method
    # traverse_and_process(root_dir)
    # Define the single file you want to process
    test_file_path = "./wiki_data/GR/wiki_01"  # Replace with the actual file path
    
    # Run process_file directly
    result = process_file(test_file_path)
    print(result) 