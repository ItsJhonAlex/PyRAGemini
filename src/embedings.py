import os
from typing import List, Union
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

class EmbeddingsGenerator:
    """
    A class to handle document embeddings generation using Google's Generative AI.
    
    Attributes:
        model_name (str): Name of the Google embedding model to use
        embeddings (GoogleGenerativeAIEmbeddings): The embedding model instance
    """
    
    def __init__(self, model_name: str = 'models/embedding-001'):
        """
        Initialize the EmbeddingsGenerator with Google's Generative AI model.
        
        Args:
            model_name (str): Name of the embedding model to use
        
        Raises:
            ValueError: If GEMINI_API_KEY is not set in environment variables
        """
        load_dotenv()
        self.google_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.google_api_key:
            raise ValueError("GEMINI_API_KEY no encontrada en las variables de entorno")
        
        self.model_name = model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=self.google_api_key
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text string.
        
        Args:
            text (str): The text to generate embeddings for
            
        Returns:
            List[float]: The generated embedding vector
            
        Raises:
            ValueError: If text is empty or not a string
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("El texto debe ser una cadena no vacía")
        
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"Error al generar embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains invalid items
        """
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            raise ValueError("La lista de textos debe contener cadenas no vacías")
        
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise Exception(f"Error al generar embeddings en lote: {str(e)}")
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of LangChain Document objects.
        
        Args:
            documents (List[Document]): List of LangChain Document objects
            
        Returns:
            List[List[float]]: List of embedding vectors
            
        Raises:
            ValueError: If documents list is empty or invalid
        """
        if not documents or not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("Se requiere una lista válida de objetos Document")
        
        try:
            texts = [doc.page_content for doc in documents]
            return self.generate_embeddings_batch(texts)
        except Exception as e:
            raise Exception(f"Error al generar embeddings para documentos: {str(e)}")


def create_embeddings_model(model_name: str = 'models/embedding-001') -> EmbeddingsGenerator:
    """
    Factory function to create an EmbeddingsGenerator instance.
    
    Args:
        model_name (str): Name of the embedding model to use
        
    Returns:
        EmbeddingsGenerator: An initialized EmbeddingsGenerator instance
    """
    return EmbeddingsGenerator(model_name)


if __name__ == '__main__':
    # Ejemplo de uso
    try:
        # Crear instancia del generador de embeddings
        embeddings_generator = create_embeddings_model()
        
        # Ejemplo con un solo texto
        texto_ejemplo = "Este es un ejemplo de texto para generar embeddings."
        embedding = embeddings_generator.generate_embedding(texto_ejemplo)
        print(f"Tamaño del embedding individual: {len(embedding)}")
        
        # Ejemplo con múltiples textos
        textos_ejemplo = [
            "Primer texto de ejemplo",
            "Segundo texto de ejemplo",
            "Tercer texto de ejemplo"
        ]
        embeddings_batch = embeddings_generator.generate_embeddings_batch(textos_ejemplo)
        print(f"Número de embeddings generados en lote: {len(embeddings_batch)}")
        print(f"Tamaño de cada embedding: {len(embeddings_batch[0])}")
        
    except Exception as e:
        print(f"Error en el ejemplo: {str(e)}")