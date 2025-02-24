# src/summarizer.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GoogleGenerativeAI
from src.document_loader import load_documents_from_dir

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configura la API de Gemini
genai.configure(api_key=google_api_key)

# Inicializa el modelo Gemini
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key, temperature=0.3)

# Modifica el prompt template para generar resúmenes
prompt_template_resumen = """
Por favor, genera un resumen conciso del siguiente texto:

TEXTO: {text}
"""
prompt_resumen = PromptTemplate(template=prompt_template_resumen, input_variables=["text"])

# Crea una cadena LLMChain para resúmenes
llm_chain_resumen = LLMChain(prompt=prompt_resumen, llm=llm)

def generate_summary(text):
    """Genera un resumen de un texto utilizando Gemini."""
    summary = llm_chain_resumen.run(text=text)
    return summary

if __name__ == '__main__':
    # Ejemplo de uso
    # Cargar documentos
    documents = load_documents_from_dir("data/documents")
    if not documents:
        print("No se encontraron documentos. Asegúrate de tener archivos en data/documents.")
        exit()

    # Unir todo el contenido en un solo texto
    texto_completo = "\n".join([doc.page_content for doc in documents])

    # Generar el resumen
    resumen = generate_summary(texto_completo)
    print(f"Resumen:\n{resumen}")