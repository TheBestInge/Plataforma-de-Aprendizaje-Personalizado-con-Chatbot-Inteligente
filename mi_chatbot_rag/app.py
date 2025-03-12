import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Verificar que la API key de OpenAI esté configurada
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: No se encontró la API key de OpenAI.")
    print("Por favor, crea un archivo .env en la raíz del proyecto con el siguiente contenido:")
    print("OPENAI_API_KEY=tu-api-key-aquí")
    sys.exit(1)

from src.indexer import DocumentIndexer
from src.retriever import InformationRetriever
from src.chatbot import RagChatbot

def main():
    """
    Función principal que ejecuta el chatbot en la línea de comandos.
    """
    # Directorios para datos e índices
    data_dir = "./data"
    persist_dir = "./storage"
    
    # Verificar que exista el directorio de datos
    data_path = Path(data_dir)
    if not data_path.exists():
        data_path.mkdir(parents=True)
        print(f"Se creó el directorio {data_dir}")
        print(f"Por favor, añade documentos en {data_dir} antes de continuar.")
        return
    
    if not any(data_path.iterdir()):
        print(f"El directorio {data_dir} está vacío. Por favor, añade documentos.")
        return
    
    print("\n===== Inicializando el chatbot RAG =====\n")
    
    # Crear o cargar el índice
    indexer = DocumentIndexer(data_dir=data_dir)
    
    try:
        if Path(persist_dir).exists():
            print("Cargando índice existente...")
            index = indexer.load_index(persist_dir=persist_dir)
        else:
            print("Creando nuevo índice de documentos...")
            index = indexer.create_index()
            indexer.save_index(persist_dir=persist_dir)
    except Exception as e:
        print(f"Error al procesar el índice: {e}")
        return
    
    # Crear el recuperador
    retriever = InformationRetriever(index=index, top_k=3)
    
    # Inicializar el chatbot
    chatbot = RagChatbot(retriever=retriever)
    
    print("\n¡Chatbot listo! Escribe 'salir' para terminar.\n")
    
    # Bucle de chat
    while True:
        try:
            user_input = input("\033[1;34mTú:\033[0m ")
            
            if user_input.lower() in ["salir", "exit", "quit", "q"]:
                print("\n¡Hasta luego! 👋")
                break
                
            print("\033[1;32mChatbot:\033[0m", end=" ")
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"\n\033[1;31mError:\033[0m {e}")

if __name__ == "__main__":
    main()