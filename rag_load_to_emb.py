import json
import os
import shutil
import sys

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "rag_data_documents"
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "startup_legal_compliance"
VECTORSTORE_CONFIG_PATH = os.path.join(CHROMA_DB_PATH, "rag_config.json")
EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data directory '{DATA_PATH}' not found.")
        print(f"Make sure your PDF files are inside a folder named '{DATA_PATH}'.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: No PDF files found in '{DATA_PATH}'.")
        sys.exit(1)

    if os.path.exists(CHROMA_DB_PATH):
        answer = input(
            f"Chroma DB already exists at '{CHROMA_DB_PATH}'. Rebuild? [y/n]: "
        ).strip().lower()
        if answer != "y":
            print("Skipping rebuild. Existing index kept.")
            sys.exit(0)
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)

    print(f"\nLoading PDFs from '{DATA_PATH}'...")
    documents = []
    for file in sorted(pdf_files):
        full_path = os.path.join(DATA_PATH, file)
        try:
            loader = PyMuPDFLoader(full_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  {file}: {len(docs)} pages")
        except Exception as e:
            print(f"  WARNING: Could not load {file}: {e}")

    if not documents:
        print("ERROR: No documents were loaded successfully.")
        sys.exit(1)

    print(f"\nTotal pages loaded: {len(documents)}")

    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        print("Check your internet connection or ensure the model is cached locally.")
        sys.exit(1)

    print("\nBuilding persisted Chroma vector database...")
    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_DB_PATH,
        )
    except Exception as e:
        print(f"ERROR: Failed to build Chroma DB: {e}")
        sys.exit(1)

    abs_path = os.path.abspath(CHROMA_DB_PATH)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    with open(VECTORSTORE_CONFIG_PATH, "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "collection_name": CHROMA_COLLECTION_NAME,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
            },
            config_file,
            indent=2,
        )

    print(f"\nChroma DB saved to: {abs_path}")
    print(f"Collection name: {CHROMA_COLLECTION_NAME}")
    print("Persistence is handled automatically by Chroma.")
    print("\nDone! You can now run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
