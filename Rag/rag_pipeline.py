import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from google import genai 
import os 
import json 
import logging
from tqdm import tqdm
from pathlib import Path 
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.Client(api_key = API_KEY)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME= "movies_db"
chroma_collection = None
is_rag_initialized = False
raw_text = []
pdf_dir = "Data/Movie_description.pdf"


def pdf_parser(pdf_dir):
    """
    Parse pdf using Pypdfloader so that it can convert pdf into texts. 
    Args: 
        pdf_dir: location of the pdf file 
    Returns: 
        pages = list of pages of the extracted text form pdf
    
    """
    reader = PyPDFLoader(pdf_dir)
    pages = reader.load()
    full_text = "\n".join([page.page_content for page in pages])
    return full_text

def split_text_to_chunks(texts, chunk_size = 1000, chunk_overlap=200):
    """
    Splitting large chunks of pdf into small manageable chunks
    Args: 
        texts: pdf in this case
        chunks_size: Size that the pdf chunks need to split into
        chunk_overlap: Overlapping amount between the chunks size
    Retuns: 
        chunks: splitted texts into different chunks size
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(texts)
    return chunks

def chunks_text(text):
    chunks = []
    movies = text.split("Title:")[1:]   
    for movie in movies:
        chunks.append("Title:"+movie.strip())
    return chunks

def batch_embed_chunks(chunks, embedding_model):
    """Embed chunks in batches."""
    try:
        embeddings = embedding_model.encode(chunks)
        return embeddings.tolist()
    except Exception as e:
         logging.error(f"Error during batch embedding: {e}")
         return None
    
def initialize_rag_database(pdf_path):
    global chroma_collection, is_rag_initialized
    if is_rag_initialized:
        print("RAG database already initialized.")
        return True

    print("--- Initializing RAG Database ---")
    try:
        text = pdf_parser(pdf_path)
        if not text:
            print("ERROR: PDF parsing yielded no text.")
            return False
        chunks = chunks_text(text)
        if not chunks:
            print("ERROR: Text chunking yielded no chunks.")
            return False
        embeddings = batch_embed_chunks(chunks, embedding_model)
        if embeddings is None: 
            print("ERROR: Embedding failed.")
            return False
        print(f"Creating ChromaDB client and collection '{COLLECTION_NAME}' (in-memory)...")
        client = chromadb.Client()
        try:
             print(f"Attempting to delete existing collection '{COLLECTION_NAME}' (if any)...")
             client.delete_collection(COLLECTION_NAME)
             print(f"Collection '{COLLECTION_NAME}' deleted.")
        except Exception as e:
             print(f"Info: Could not delete collection '{COLLECTION_NAME}' (may not exist yet): {e}")

        print(f"Creating collection '{COLLECTION_NAME}'...")
        collection = client.create_collection(COLLECTION_NAME)
        print("Collection created.")
        print(f"Adding {len(chunks)} documents to collection...")
        ids = [str(i) for i in range(len(chunks))]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks
        )
        print("Documents added.")

        chroma_collection = collection 
        is_rag_initialized = True
        print("--- RAG Database Initialization SUCCESSFUL ---")
        return True

    except Exception as e:
        print(f"--- RAG Database Initialization FAILED ---")
        logging.error(f"Failed to initialize RAG database: {e}", exc_info=True)
        print(f"ERROR: {e}")
        is_rag_initialized = False
        chroma_collection = None
        return False

def query_rag_database(query_text, n_results=3):
    """Queries the globally initialized ChromaDB collection."""
    global chroma_collection, is_rag_initialized
    if not is_rag_initialized or chroma_collection is None:
        print("ERROR: RAG database is not initialized. Cannot query.")
        return [] 

    print(f"\nQuerying RAG database for: '{query_text}'")
    try:
        query_embedding = embedding_model.encode(query_text).tolist()
        results = chroma_collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=['documents'] 
        )
        retrieved_docs = results.get('documents', [[]])[0] 
        print(f"Retrieved {len(retrieved_docs)} documents from RAG.")
        return retrieved_docs
    except Exception as e:
        print(f"ERROR during RAG query: {e}")
        logging.error(f"Failed to query RAG database: {e}", exc_info=True)
        return []


