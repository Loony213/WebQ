# Importa la función load_dotenv del módulo dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importa el módulo os para interactuar con el sistema operativo
import os
# Importa la clase PdfReader del módulo PyPDF2 para leer archivos PDF
from PyPDF2 import PdfReader
# Importa la biblioteca Streamlit para crear aplicaciones web interactivas
import streamlit as st
# Importa el CharacterTextSplitter del módulo langchain.text_splitter para dividir texto en caracteres
from langchain.text_splitter import CharacterTextSplitter
# Importa OpenAIEmbeddings del módulo langchain.embeddings.openai para generar incrustaciones de texto utilizando OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
# Importa FAISS del módulo langchain para realizar búsqueda de similitud
from langchain import FAISS
# Importa load_qa_chain del módulo langchain.chains.question_answering para cargar cadenas de preguntas y respuestas
from langchain.chains.question_answering import load_qa_chain
# Importa OpenAI del módulo langchain.llms para interactuar con el modelo de lenguaje de OpenAI
from langchain.llms import OpenAI
# Importa get_openai_callback del módulo langchain.callbacks para obtener realimentación de OpenAI
from langchain.callbacks import get_openai_callback
# Importa el módulo langchain
import langchain
# Importa la biblioteca requests para hacer solicitudes HTTP
import requests

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de una URL
def process_url(url):
    # Realiza una solicitud HTTP para obtener el contenido de la URL
    response = requests.get(url)
    # Extrae el texto del contenido de la respuesta
    text = response.text
    # Procesa el texto
    knowledgeBase = process_text(text)
    return knowledgeBase

# Función para procesar el texto extraído de un archivo PDF
def process_text(text):
  # Divide el texto en trozos usando langchain
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len
  )

  chunks = text_splitter.split_text(text)

  # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
  embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

  knowledge_base = FAISS.from_texts(chunks, embeddings)

  return knowledge_base

# Función principal de la aplicación
def main():
  st.title("Preguntas a un PDF")  # Establece el título de la aplicación

  # Caja de entrada de texto para que el usuario escriba la URL
  url = st.text_input('Escribe la URL de la página web...')

  if url:
    # Procesa la URL y obtiene la base de conocimientos
    knowledgeBase = process_url(url)

    # Caja de entrada de texto para que el usuario escriba su pregunta
    query = st.text_input('Escribe tu pregunta para la página web...')

    # Botón para cancelar la pregunta
    cancel_button = st.button('Cancelar')

    if cancel_button:
      st.stop()  # Detiene la ejecución de la aplicación

    if query:
      # Realiza una búsqueda de similitud en la base de conocimientos
      docs = knowledgeBase.similarity_search(query)

      # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
      model = "gpt-3.5-turbo-instruct" # Acepta 4096 tokens
      temperature = 0  # Valores entre 0 - 1
      llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

      # Carga la cadena de preguntas y respuestas
      chain = load_qa_chain(llm, chain_type="map_reduce")

      # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
      with get_openai_callback() as cost:
        response = chain.invoke(input={"question": query, "input_documents": docs})
        st.write("El costo de la operación es:", cost)
        st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
  main()  # Llama a la función principal
