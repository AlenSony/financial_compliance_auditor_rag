import os
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
# pyrefly: ignore [missing-import]
from langchain_ollama import OllamaEmbeddings
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_financial_pdf():
    print("Processing Financial PDF...")
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")

    pdf_dir = "target_documents"

    if not os.path.exists(pdf_dir):
        os.mkdir(pdf_dir)
        print(f"Successfully created '{pdf_dir}' directory. Place a financial pdf in it and try again")
        return
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'. Please add a financial PDF and try again.")
        return
    target_path = os.path.join(pdf_dir, pdf_files[0])
    print(f"Loading '{target_path}'...")

    loader = PyPDFLoader(target_path)
    raw_pages = loader.load()

    print(f"Successfully read {len(raw_pages)} pages from the PDF. Starting text splitting...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500, chunk_overlap = 300
        )
    
    processed_chunks = text_splitter.split_documents(raw_pages)
    print(f"Successfully split {len(processed_chunks)} chunks from the PDF.")

    print("Embedding chunks and creating vector store...")
    Chroma.from_documents(
        documents=processed_chunks,
        embedding=embeddings,
        persist_directory="audit_db"
    )

    
    print("✅ Successfully processed the PDF and created the vector store.")
    
if __name__ == "__main__":
    process_financial_pdf() 
    
    