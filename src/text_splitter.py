from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks for better processing
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of split documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    # Test functionality
    from document_loader import load_documents_from_dir
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    documents_dir = os.path.join(project_root, "data", "documents")
    
    documents = load_documents_from_dir(documents_dir)
    if documents:
        split_documents_result = split_documents(documents)
        print(f"Original documents: {len(documents)}")
        print(f"Split documents: {len(split_documents_result)}")
        print(f"First chunk content: {split_documents_result[0].page_content[:100]}...")