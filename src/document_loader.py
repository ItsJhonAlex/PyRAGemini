from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
import os
from typing import List, Optional
from langchain.schema import Document

def load_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If file format is not supported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        
    return loader.load()

def load_documents_from_dir(directory: str) -> List[Document]:
    """
    Load all supported documents from a directory
    
    Args:
        directory: Path to the directory containing documents
        
    Returns:
        List of Document objects
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    documents = []
    supported_extensions = {'.pdf', '.docx', '.xlsx', '.txt'}
    
    for filename in os.listdir(directory):
        fpath = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext in supported_extensions:
            try:
                documents.extend(load_document(fpath))
            except Exception as e:
                print(f"Error loading {fpath}: {str(e)}")
                
    return documents

#Ejemplo de uso
if __name__ == "__main__":
    # Construir la ruta absoluta al directorio de documentos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    documents_dir = os.path.join(project_root, "data", "documents")
    
    try:
        documents = load_documents_from_dir(documents_dir)
        if documents:
            print(f"Loaded {len(documents)} documents successfully")
            print(f"First document preview: {documents[0].page_content[:100]}...")
        else:
            print("No documents found")
    except Exception as e:
        print(f"Error: {str(e)}")