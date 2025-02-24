import os
from typing import List, Optional
from src.document_loader import load_documents_from_dir
from src.text_splitter import split_documents
from src.embedings import create_embeddings_model
from src.indexing import IndexManager  # Corregida la importación
from langchain.schema import Document
from dotenv import load_dotenv

def process_documents(
    documents_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Optional[List[Document]]:
    """
    Process documents from a directory through the RAG pipeline
    
    Args:
        documents_dir: Directory containing the documents
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        Optional[List[Document]]: List of processed document chunks or None if no documents found
    """
    print(f"\n1. Loading documents from {documents_dir}")
    documents = load_documents_from_dir(documents_dir)
    if not documents:
        print("No documents found!")
        return None
    
    print(f"Found {len(documents)} documents")
    
    print("\n2. Splitting documents into chunks")
    split_docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Created {len(split_docs)} chunks")
    print(f"Sample chunk: {split_docs[0].page_content[:200]}...")
    
    return split_docs

def generate_embeddings_for_chunks(chunks: List[Document]) -> Optional[List[List[float]]]:
    """
    Generate embeddings for document chunks using Google's Generative AI
    
    Args:
        chunks: List of document chunks to generate embeddings for
        
    Returns:
        Optional[List[List[float]]]: List of embedding vectors or None if error occurs
    """
    try:
        print("\n3. Generating embeddings for chunks")
        embeddings_generator = create_embeddings_model()
        embeddings = embeddings_generator.embed_documents(chunks)
        print(f"Generated embeddings for {len(embeddings)} chunks")
        print(f"Embedding dimension: {len(embeddings[0])}")
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None

def main():
    # Load environment variables
    load_dotenv()
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    documents_dir = os.path.join(current_dir, "data", "documents")
    
    # Process documents
    chunks = process_documents(
        documents_dir,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    if not chunks:
        print("\nError: No document chunks to process")
        return
    
    # Create index manager
    index_manager = IndexManager()
    
    # Create and save index
    try:
        print("\n3. Creating vector index")
        db = index_manager.create_index(chunks)
        index_manager.save_index(db, "documentos_index")
        
        # Test search
        print("\n4. Testing search functionality")
        query = "¿Qué es el aprendizaje automático?"
        results = index_manager.similarity_search(db, query)
        
        print(f"\nTop {len(results)} results for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        return
    
    print("\nRAG Pipeline completed successfully!")
    print(f"Total documents processed: {len(chunks)}")
    print("Vector index created and saved")
    print("Search functionality tested")

if __name__ == "__main__":
    main()