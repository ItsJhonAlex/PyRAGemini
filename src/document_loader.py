from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader
import os

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".xlsx"):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("File format not supported")
    return loader.load()

def load_documents_from_dir(directory):
    documents = []
    for filename in os.listdir(directory):
        fpath = os.path.join(directory, filename)
        try:
            documents.extend(load_document(fpath))
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
    return documents

#Ejemplo de uso
if __name__ == "__main__":
    # Construir la ruta absoluta al directorio de documentos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    documents_dir = os.path.join(project_root, "data", "documents")
    
    documents = load_documents_from_dir(documents_dir)
    if documents:
        print(f"I have {len(documents)} documents.")
        print(f"First 50 characters: {documents[0].page_content[:50]}...")
    else:
        print("No documents found.")