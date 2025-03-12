import os
from pathlib import Path
from typing import List, Optional

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

class DocumentIndexer:
    """
    Clase para indexar documentos y crear vectores de búsqueda.
    
    Esta clase maneja la carga de documentos desde un directorio,
    los procesa, y crea un índice vectorial que permite búsquedas
    semánticas en el contenido.
    """
    
    def __init__(self, data_dir: str, index_name: str = "llamaindex_chatbot"):
        """
        Inicializa el indexador de documentos.
        
        Args:
            data_dir: Directorio donde se encuentran los documentos
            index_name: Nombre del índice a crear
        """
        # Configurar los modelos de LlamaIndex
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
        
        self.data_dir = Path(data_dir)
        self.index_name = index_name
        self.index = None
    
    def load_documents(self) -> List[Document]:
        """
        Carga los documentos desde el directorio especificado.
        
        Returns:
            Lista de documentos cargados
        
        Raises:
            FileNotFoundError: Si el directorio no existe
            ValueError: Si el directorio está vacío
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"El directorio {self.data_dir} no existe")
        
        # Verificar si hay archivos en el directorio
        if not any(self.data_dir.iterdir()):
            raise ValueError(f"El directorio {self.data_dir} está vacío. Por favor, añade documentos.")
        
        # Cargar documentos con SimpleDirectoryReader
        print(f"Cargando documentos desde {self.data_dir}...")
        documents = SimpleDirectoryReader(str(self.data_dir)).load_data()
        
        print(f"Se cargaron {len(documents)} documentos")
        return documents
    
    def create_index(self) -> VectorStoreIndex:
        """
        Crea un índice de vectores a partir de los documentos cargados.
        
        Returns:
            Índice vectorial creado
        """
        documents = self.load_documents()
        
        print("Creando índice de vectores...")
        self.index = VectorStoreIndex.from_documents(documents)
        print("Índice creado exitosamente")
        
        return self.index
    
    def save_index(self, persist_dir: str = "./storage") -> None:
        """
        Guarda el índice en disco para uso futuro.
        
        Args:
            persist_dir: Directorio donde se guardará el índice
        
        Raises:
            ValueError: Si no hay un índice para guardar
        """
        if not self.index:
            raise ValueError("No hay un índice para guardar. Ejecuta create_index primero.")
        
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Guardando índice en {persist_path}...")
        self.index.storage_context.persist(persist_dir=str(persist_path))
        print(f"Índice guardado exitosamente en {persist_path}")
    
    def load_index(self, persist_dir: str = "./storage") -> Optional[VectorStoreIndex]:
        """
        Carga un índice guardado previamente.
        
        Args:
            persist_dir: Directorio desde donde cargar el índice
        
        Returns:
            Índice cargado o None si no se pudo cargar
        
        Raises:
            FileNotFoundError: Si no se encuentra un índice en el directorio
        """
        persist_path = Path(persist_dir)
        
        if not persist_path.exists():
            raise FileNotFoundError(f"No se encontró ningún índice en {persist_path}")
        
        from llama_index.core import StorageContext, load_index_from_storage
        
        print(f"Cargando índice desde {persist_path}...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        self.index = load_index_from_storage(storage_context)
        print("Índice cargado exitosamente")
        
        return self.index