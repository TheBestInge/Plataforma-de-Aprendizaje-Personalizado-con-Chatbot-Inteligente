from typing import Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.response import Response

class InformationRetriever:
    """
    Clase para recuperar información relevante del índice.
    
    Esta clase se encarga de buscar en el índice y encontrar
    la información más relevante para responder a las consultas
    del usuario.
    """
    
    def __init__(self, index: VectorStoreIndex, top_k: int = 3):
        """
        Inicializa el recuperador de información.
        
        Args:
            index: Índice de vectores desde el cual recuperar información
            top_k: Número de resultados más relevantes a recuperar
        """
        self.index = index
        self.top_k = top_k
        self.retriever = None
        self.query_engine = None
        self._setup_retriever()
    
    def _setup_retriever(self) -> None:
        """
        Configura el recuperador y el motor de consultas.
        
        Esta función establece cómo se recuperarán los documentos
        y cómo se procesarán para generar respuestas.
        """
        # Crear un recuperador que devuelve los top_k nodos más relevantes
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )
        
        # Configurar el sintetizador de respuestas (cómo se generan las respuestas)
        # CompactAndRefine primero compacta el contexto y luego refina la respuesta
        response_synthesizer = CompactAndRefine()
        
        # Configurar el motor de consultas que combina recuperación y síntesis
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer
        )
    
    def retrieve_information(self, query_text: str) -> Response:
        """
        Recupera información relevante basada en la consulta del usuario.
        
        Args:
            query_text: Texto de la consulta del usuario
        
        Returns:
            Respuesta generada basada en la información recuperada
        
        Raises:
            ValueError: Si el motor de consultas no está configurado
        """
        if not self.query_engine:
            raise ValueError("El motor de consultas no está configurado")
        
        # Procesar la consulta y obtener una respuesta
        response = self.query_engine.query(query_text)
        return response
    
    def get_retrieval_context(self, query_text: str) -> Dict[str, Any]:
        """
        Obtiene el contexto de recuperación para depuración o análisis.
        
        Args:
            query_text: Texto de la consulta
        
        Returns:
            Diccionario con información sobre los documentos recuperados
        """
        if not self.retriever:
            raise ValueError("El recuperador no está configurado")
        
        # Recuperar nodos relevantes
        nodes = self.retriever.retrieve(query_text)
        
        # Crear un resumen de los nodos recuperados
        context_info = {
            "num_nodes_retrieved": len(nodes),
            "nodes_summary": [
                {
                    "score": node.score if hasattr(node, "score") else None,
                    "text_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                }
                for node in nodes
            ]
        }
        
        return context_info