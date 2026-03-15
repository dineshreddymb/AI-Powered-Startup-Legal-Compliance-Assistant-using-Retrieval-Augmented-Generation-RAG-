import os
import sys

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "rag_data_documents"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main():
    # Validate data directory
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data directory '{DATA_PATH}' not found.")
        print(f"Make sure your PDF files are inside a folder named '{DATA_PATH}'.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: No PDF files found in '{DATA_PATH}'.")
        sys.exit(1)

    # Check if index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        answer = input(
            f"FAISS index already exists at '{FAISS_INDEX_PATH}'. Rebuild? [y/n]: "
        ).strip().lower()
        if answer != "y":
            print("Skipping rebuild. Existing index kept.")
            sys.exit(0)

    # Step 1: Load all PDFs
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

    # Step 2: Split into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # Step 3: Load embedding model (downloads ~90MB on first run)
    print(f"\nLoading embedding model '{EMBEDDING_MODEL}'...")
    print("(This may take a minute on first run — model will be cached locally.)")
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        print("Check your internet connection or set HF_HUB_OFFLINE=1 if using cached model.")
        sys.exit(1)

    # Step 4: Build FAISS index
    print("\nBuilding FAISS vector index...")
    try:
        vectordb = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    except Exception as e:
        print(f"ERROR: Failed to build FAISS index: {e}")
        sys.exit(1)

    # Step 5: Save index
    vectordb.save_local(FAISS_INDEX_PATH)
    abs_path = os.path.abspath(FAISS_INDEX_PATH)
    print(f"\nFAISS index saved to: {abs_path}")
    print("  - faiss_index/index.faiss")
    print("  - faiss_index/index.pkl")
    print("\nDone! You can now run the Streamlit app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
