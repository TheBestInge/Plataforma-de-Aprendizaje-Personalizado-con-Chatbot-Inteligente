import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

from src.indexer import DocumentIndexer
from src.retriever import InformationRetriever
from src.chatbot import RagChatbot

# Configuración de la página
st.set_page_config(
    page_title="Chatbot RAG con LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Configurar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .assistant-message {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .file-uploader {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown("<h1 class='main-header'>Chatbot RAG con LlamaIndex 🦙</h1>", unsafe_allow_html=True)
st.markdown("Este chatbot responde preguntas basándose en los documentos que has subido. Utiliza LlamaIndex para el procesamiento de documentos y la generación de respuestas.")

# Inicializar variables de estado de la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "index_created" not in st.session_state:
    st.session_state.index_created = False

# Sidebar para configuración
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Configuración</h2>", unsafe_allow_html=True)
    
    # API Key de OpenAI
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Introduce tu API key de OpenAI"
    )
    
    # Guardar API key en variables de entorno
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Directorio de datos
    data_dir = "./data"
    st.text_input("Directorio de datos", value=data_dir, disabled=True)
    
    # Sección para subir archivos
    st.markdown("<h3>Subir documentos</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Sube documentos para entrenar el chatbot",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md"],
        help="Puedes subir múltiples archivos. Formatos soportados: PDF, TXT, DOCX, MD"
    )
    
    # Guardar archivos subidos
    if uploaded_files:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        
        with st.spinner("Guardando archivos..."):
            for file in uploaded_files:
                file_path = data_path / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            st.success(f"{len(uploaded_files)} archivos guardados en {data_dir}")
            # Invalidar el índice cuando se suben nuevos archivos
            st.session_state.index_created = False
    
    # Botón para crear/recargar índice
    create_index_btn = st.button(
        "Crear/Recargar Índice",
        help="Crea un nuevo índice o recarga el existente con los documentos actuales",
        use_container_width=True
    )
    
    if create_index_btn:
        if not api_key:
            st.error("Por favor, introduce tu API key de OpenAI")
        else:
            with st.spinner("Creando índice..."):
                try:
                    # Crear el indexador y procesar documentos
                    indexer = DocumentIndexer(data_dir=data_dir)
                    index = indexer.create_index()
                    indexer.save_index(persist_dir="./storage")
                    
                    # Crear el recuperador
                    retriever = InformationRetriever(index=index, top_k=3)
                    
                    # Inicializar el chatbot
                    st.session_state.chatbot = RagChatbot(retriever=retriever)
                    st.session_state.index_created = True
                    
                    st.success("¡Índice creado con éxito!")
                except Exception as e:
                    st.error(f"Error al crear el índice: {e}")
    
    # Botón para limpiar el historial de chat
    if st.button("Limpiar historial de chat", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_history()
        st.success("Historial de chat limpiado")
    
    # Información adicional
    st.markdown("---")
    st.markdown("""
    ### Sobre RAG
    **Retrieval-Augmented Generation (RAG)** es una técnica que combina:
    
    1. **Recuperación** de información relevante de documentos
    2. **Generación** de respuestas basadas en esa información
    
    Esto permite que el chatbot responda con precisión usando la información de tus documentos.
    """)

# Área principal - Chatbot
# Verificar si hay documentos disponibles
data_path = Path(data_dir)
if not data_path.exists() or not any(data_path.iterdir()):
    st.warning("No hay documentos disponibles. Por favor, sube algunos documentos usando el panel lateral.")
    st.stop()

# Cargar el chatbot si no está inicializado
if st.session_state.chatbot is None and not st.session_state.index_created:
    if os.getenv("OPENAI_API_KEY"):
        with st.spinner("Cargando chatbot..."):
            try:
                # Intentar cargar un índice existente
                indexer = DocumentIndexer(data_dir=data_dir)
                index = indexer.load_index(persist_dir="./storage")
                
                # Crear el recuperador
                retriever = InformationRetriever(index=index, top_k=3)
                
                # Inicializar el chatbot
                st.session_state.chatbot = RagChatbot(retriever=retriever)
                st.session_state.index_created = True
                
                st.success("Chatbot cargado correctamente")
            except FileNotFoundError:
                st.info("No se encontró un índice existente. Por favor, crea uno con el botón 'Crear/Recargar Índice'.")
            except Exception as e:
                st.error(f"Error al cargar el chatbot: {e}")
    else:
        st.warning("Por favor, introduce tu API key de OpenAI en el panel lateral")

# Mostrar mensajes del historial
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'><b>Tú:</b> {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'><b>Asistente:</b> {message['content']}</div>", unsafe_allow_html=True)

# Entrada de chat
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    if not api_key:
        st.error("Por favor, introduce tu API key de OpenAI en el panel lateral")
    elif not st.session_state.index_created:
        st.error("Por favor, crea un índice primero usando el botón 'Crear/Recargar Índice'")
    else:
        # Agregar mensaje del usuario al historial
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='chat-message user-message'><b>Tú:</b> {prompt}</div>", unsafe_allow_html=True)
        
        # Generar y mostrar respuesta
        with st.spinner("Pensando..."):
            try:
                response = st.session_state.chatbot.chat(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(f"<div class='chat-message assistant-message'><b>Asistente:</b> {response}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error al generar respuesta: {e}")