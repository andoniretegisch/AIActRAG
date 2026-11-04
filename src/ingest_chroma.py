
import json
import numpy as np
from pathlib import Path
import chromadb as cmdb

EMBEDDINGS_PATH = Path("data/processed/embeddings_v2.npy")
METADATA_PATH = Path("data/processed/embedding_metadata_v2.json")

def get_data():
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
        
    embeddings = (np.load(EMBEDDINGS_PATH)).tolist()
    documents = [mtdt["text"] for mtdt in metadata["chunks"]]
    ids = [mtdt["chunk_id"] for mtdt in metadata["chunks"]]
    metadatas = [{key: value for key, value in mtdt.items() if key != "text" and value is not None} for mtdt in metadata["chunks"]]
    return embeddings, documents, ids, metadatas

def main():
    
    persistent_client = cmdb.PersistentClient("data/chroma_db")
    collection = persistent_client.get_or_create_collection("ai_act_chunks", metadata={"hnsw:space":"cosine"})
    embeddings, documents, ids, metadatas = get_data()
    collection.add(ids, embeddings, metadatas, documents)



if __name__ == "__main__":
    main()