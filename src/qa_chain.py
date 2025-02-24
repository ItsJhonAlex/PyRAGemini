# src/qa_chain.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GoogleGenerativeAI
from src.indexer import search_index  # Importa la función search_index
from src.embeddings import create_embeddings_model # Importa la funcion de embeddings

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configura la API de Gemini
genai.configure(api_key=google_api_key)

# Inicializa el modelo Gemini
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.3)

# Crea un prompt template para la generación de respuestas
prompt_template = """
Utiliza el siguiente contexto para responder la pregunta al final.
Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

CONTEXTO: {context}

PREGUNTA: {question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Crea la cadena LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

def generate_answer(query, db, k=4):
    """
    Genera una respuesta a una pregunta utilizando Gemini, buscando primero en el índice FAISS.
    Args:
        query: La pregunta a responder.
        db: El índice FAISS.
        k: Número de documentos a recuperar del índice.
    Returns:
        La respuesta generada por Gemini.
    """
    docs = search_index(db, query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    response = llm_chain.run(context=context, question=query)
    return response

if __name__ == '__main__':
    # Ejemplo de uso (necesitas cargar documentos, fragmentarlos y crear el índice primero)
    from src import document_loader, text_splitter, indexer

    # Cargar documentos
    documents = document_loader.load_documents_from_dir("data/documents")
    if not documents:
        print("No se encontraron documentos. Asegúrate de tener archivos en data/documents.")
        exit()

    # Fragmentar el texto
    fragments = text_splitter.split_text(documents)

    # Crear el índice
    embeddings = create_embeddings_model()
    db = indexer.create_index(fragments, embeddings)

    # Realizar una pregunta
    pregunta = "¿De qué trata este documento?"
    respuesta = generate_answer(pregunta, db)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {respuesta}")