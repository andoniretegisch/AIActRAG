import retrieve_chroma as rtrv
import llm
from sentence_transformers import SentenceTransformer
import chromadb as cmdb
import generate

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def run_pipeline(query:str, model, collection):
    results = rtrv.retrieve(query, model, collection)
    chunks = rtrv.format_results(results)
    prompt = generate.prepare_prompt(query, chunks)
    answer = llm.generate_answer(prompt)

    return answer


def main():
    model = SentenceTransformer(MODEL_NAME)
    persistent_client = cmdb.PersistentClient(path="data/chroma_db")
    collection = persistent_client.get_collection("ai_act_chunks")
    query = input("What do you want to know about the AI Act?")
    answer = run_pipeline(query, model, collection)
    print("Answer: " + answer)
    

if __name__ == "__main__":
    main()
