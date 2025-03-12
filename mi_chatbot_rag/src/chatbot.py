from typing import List, Dict, Any

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from src.retriever import InformationRetriever

class RagChatbot:
    """
    Chatbot basado en Retrieval-Augmented Generation (RAG).
    
    Esta clase implementa un chatbot que utiliza un sistema RAG
    para responder preguntas basándose en la información de documentos.
    """
    
    def __init__(self, retriever: InformationRetriever):
        """
        Inicializa el chatbot RAG.
        
        Args:
            retriever: Recuperador de información configurado
        """
        self.retriever = retriever
        # Crear un buffer de memoria para almacenar el historial de chat
        self.chat_history = ChatMemoryBuffer.from_defaults(token_limit=4096)
        
        # Prompt del sistema que define el comportamiento del chatbot
        self.system_prompt = """
        Eres un asistente útil y experto basado en RAG (Retrieval-Augmented Generation).
        
        Tu tarea es responder preguntas basándote únicamente en la información proporcionada 
        en tu contexto. No inventes información ni uses conocimiento que no esté en el contexto.
        
        Si no conoces la respuesta o la información no está en el contexto proporcionado, 
        indica amablemente que no tienes esa información y sugiere al usuario que reformule
        su pregunta o proporcione más documentos con esa información.
        
        Utiliza un tono profesional y amigable. Organiza tus respuestas de forma clara y concisa.
        """
        
        # Configurar el modelo de lenguaje para generar respuestas
        self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    def chat(self, user_input: str) -> str:
        """
        Procesa la entrada del usuario y genera una respuesta.
        
        Args:
            user_input: Pregunta o mensaje del usuario
        
        Returns:
            Respuesta generada por el chatbot
        """
        # Recuperar información relevante de los documentos
        retrieval_response = self.retriever.retrieve_information(user_input)
        
        # Obtener el historial de chat
        chat_history = self._format_chat_history()
        
        # Crear un prompt que combina el contexto, historial y la pregunta
        context = str(retrieval_response)
        prompt = f"""
        {self.system_prompt}
        
        Historial de la conversación:
        {chat_history}
        
        Contexto recuperado de los documentos:
        {context}
        
        Pregunta del usuario: {user_input}
        
        Por favor, responde basándote únicamente en el contexto proporcionado.
        """
        
        # Generar la respuesta
        response = self.llm.complete(prompt)
        
        # Actualizar el historial de chat
        self.chat_history.put("user", user_input)
        self.chat_history.put("assistant", str(response))
        
        return str(response)
    
    def _format_chat_history(self) -> str:
        """
        Formatea el historial de chat para incluirlo en el prompt.
        
        Returns:
            Historial de chat formateado como texto
        """
        messages = self.chat_history.get_messages()
        if not messages:
            return "No hay historial de chat previo."
        
        formatted_history = []
        for msg in messages:
            role = "Usuario" if msg.role == "user" else "Asistente"
            formatted_history.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted_history)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Devuelve el historial completo de la conversación.
        
        Returns:
            Lista de mensajes en el historial
        """
        messages = self.chat_history.get_messages()
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def clear_history(self) -> None:
        """
        Limpia el historial de chat.
        """
        self.chat_history.reset()