# src/indexing.py
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.embedings import create_embeddings_model

class IndexManager:
    """
    Clase para manejar la creación, guardado y carga de índices FAISS.
    
    Attributes:
        index_dir (str): Directorio donde se guardarán los índices
        embeddings_model: Modelo de embeddings a utilizar
    """
    
    def __init__(self, index_dir: str = "indexes"):
        """
        Inicializa el IndexManager.
        
        Args:
            index_dir: Directorio donde se guardarán los índices
        """
        self.index_dir = index_dir
        self.embeddings_model = create_embeddings_model()
        
        # Crear directorio de índices si no existe
        os.makedirs(self.index_dir, exist_ok=True)
    
    def create_index(self, chunks: List[Document]) -> FAISS:
        """
        Crea un índice FAISS a partir de fragmentos de documento.
        
        Args:
            chunks: Lista de fragmentos de documento
            
        Returns:
            FAISS: Índice vectorial creado
            
        Raises:
            ValueError: Si no se proporcionan chunks
        """
        if not chunks:
            raise ValueError("No se proporcionaron documentos para indexar")
        
        try:
            print(f"\nCreando índice con {len(chunks)} fragmentos...")
            db = FAISS.from_documents(chunks, self.embeddings_model)
            print("Índice creado exitosamente")
            return db
        except Exception as e:
            raise Exception(f"Error al crear el índice: {str(e)}")
    
    def save_index(self, db: FAISS, index_name: str) -> None:
        """
        Guarda un índice FAISS en disco.
        
        Args:
            db: Índice FAISS a guardar
            index_name: Nombre del índice
            
        Raises:
            ValueError: Si no se proporciona un índice válido o nombre
        """
        if not db or not index_name:
            raise ValueError("Se requiere un índice válido y un nombre")
        
        try:
            index_path = os.path.join(self.index_dir, index_name)
            db.save_local(index_path)
            print(f"Índice guardado en: {index_path}")
        except Exception as e:
            raise Exception(f"Error al guardar el índice: {str(e)}")
    
    def load_index(self, index_name: str) -> Optional[FAISS]:
        """
        Carga un índice FAISS desde disco.
        
        Args:
            index_name: Nombre del índice a cargar
            
        Returns:
            FAISS: Índice cargado o None si no existe
            
        Raises:
            FileNotFoundError: Si el índice no existe
        """
        index_path = os.path.join(self.index_dir, index_name)
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el índice: {index_path}")
        
        try:
            print(f"\nCargando índice desde: {index_path}")
            db = FAISS.load_local(index_path, self.embeddings_model)
            print("Índice cargado exitosamente")
            return db
        except Exception as e:
            raise Exception(f"Error al cargar el índice: {str(e)}")
    
    def similarity_search(
        self,
        db: FAISS,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Realiza una búsqueda de similitud en el índice.
        
        Args:
            db: Índice FAISS a usar
            query: Texto de consulta
            k: Número de resultados a retornar
            
        Returns:
            List[Document]: Lista de documentos similares
            
        Raises:
            ValueError: Si no se proporciona un índice válido o consulta
        """
        if not db or not query:
            raise ValueError("Se requiere un índice válido y una consulta")
        
        try:
            results = db.similarity_search(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"Error en la búsqueda: {str(e)}")


if __name__ == '__main__':
    # Ejemplo de uso
    from src.document_loader import load_documents_from_dir
    from src.text_splitter import split_documents
    
    try:
        # Configurar rutas
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        documents_dir = os.path.join(project_root, "data", "documents")
        
        # Crear gestor de índices
        index_manager = IndexManager()
        
        # Cargar y procesar documentos
        print("\n1. Cargando documentos...")
        documents = load_documents_from_dir(documents_dir)
        if not documents:
            print("No se encontraron documentos para indexar")
            exit(1)
            
        print(f"Se cargaron {len(documents)} documentos")
        
        # Dividir documentos en chunks
        print("\n2. Dividiendo documentos en fragmentos...")
        chunks = split_documents(documents)
        print(f"Se crearon {len(chunks)} fragmentos")
        
        # Crear y guardar índice
        print("\n3. Creando índice vectorial...")
        db = index_manager.create_index(chunks)
        index_manager.save_index(db, "documentos_index")
        
        # Probar búsqueda
        print("\n4. Probando búsqueda...")
        query = "Ejemplo de consulta"
        results = index_manager.similarity_search(db, query)
        print(f"\nResultados para la consulta: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nResultado {i}:")
            print(f"Contenido: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"\nError: {str(e)}")