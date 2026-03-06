import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
import hashlib
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def ingest_dataframe_info(df: pd.DataFrame):
    # FIXED: Safely fetch existing IDs and delete them
    existing_data = collection.get()
    if existing_data and existing_data['ids']:
        collection.delete(ids=existing_data['ids'])
        
    documents = []
    
    for col in df.columns:
        documents.append(f"Column '{col}': {df[col].dtype} type, {df[col].nunique()} unique, {df[col].isna().sum()} missing")
    
    for col in df.select_dtypes(include=[np.number]).columns:
        documents.append(f"Stats '{col}': min={df[col].min()}, max={df[col].max()}, mean={df[col].mean()}")
        
    embeddings = embedder.encode(documents).tolist()
    ids = [hashlib.md5(text.encode()).hexdigest()[:12] for text in documents]
    
    if documents:
        collection.add(documents=documents, embeddings=embeddings, ids=ids)

def retrieve_context(query: str, k: int = 5) -> str:
    try:
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=k)
        if not results["documents"] or not results["documents"][0]: return ""
        return "\n".join(results["documents"][0])
    except:
        return ""