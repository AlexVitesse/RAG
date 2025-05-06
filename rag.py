# Importaciones mejor organizadas
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
from functools import lru_cache

# LangChain
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Contador de tokens
import tiktoken

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def count_tokens(text: str, model_name: str = "llama3") -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def costo_tokens(text: str, type: str = "entrada") -> float:
    """Calcula el costo de los tokens para el modelo Llama 3."""
    num_tokens = count_tokens(text)
    if type == "entrada":
        return num_tokens * (0.05/1000000)
    elif type == "salida":
        return num_tokens * (0.00000008)
    else:
        raise ValueError("Modelo no soportado para cálculo de costos.")

class RAGSystem:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self._init_components()
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def _init_components(self):
        """Inicializa todos los componentes del sistema RAG"""
        self.llm = self._setup_llm()
        self.embed_model = self._setup_embeddings()
        self.vectorstore = self._load_vectorstore()
        self.retriever = self._setup_retriever()
        self.qa_chain = self._setup_qa_chain()
    
    @lru_cache(maxsize=1)  # Cache para evitar cargas múltiples
    def _load_vectorstore(self):
        """Carga la base de datos vectorial existente"""
        chroma_dir = "chroma_db_dir"
        if not Path(chroma_dir).exists():
            raise FileNotFoundError(
                f"Directorio de base vectorial no encontrado: {chroma_dir}. "
                "Por favor carga los documentos primero usando vectorial.py"
            )
        
        logger.info("✅ Cargando base vectorial existente desde caché...")
        vectorstore = Chroma(
            embedding_function=self.embed_model,
            persist_directory=chroma_dir,
            collection_name="papalote"
        )
        
        # Cargamos los documentos solo para el BM25Retriever
        self.docs = list(vectorstore.get()['documents'])
        
        return vectorstore
    
    def _setup_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name="llama3-8b-8192",
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=self.groq_api_key,
            temperature=0.8,
            top_p=0.9,
            presence_penalty=0.2,
        )
    
    def _setup_embeddings(self) -> FastEmbedEmbeddings:
        return FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            max_length=384
        )
    
    def _setup_retriever(self):
        vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_texts(self.docs)
        bm25_retriever.k = 10
        
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.4, 0.6]
        )
    
    def _setup_qa_chain(self):
        prompt_template = """
        Eres un guía experto del Museo Papalote con dos roles:
        1. Cuando la pregunta es sobre ciencia o el museo: Responde como un educador entusiasta para niños.
        2. Para otras preguntas: Sé un asistente amable que redirige al tema del museo.

        Contexto del Museo Papalote (usa esto si es relevante):
        {context}

        Instrucciones específicas:
        - Lenguaje: Sencillo y divertido (como explicar a un niño de 8 años)
        - Estructura: Máximo 3 oraciones + imagen mágica final (emoji/analogía)
        - Técnicas: Usa comparaciones con animales, superhéroes o juguetes
        - Si la pregunta no es del museo: Relaciónala amablemente con la ciencia

        Ejemplos:

        Pregunta científica:
        "¿Por qué el agua se congela?"

        Respuesta:
        "¡El agua se congela como si se pusiera su pijama de invierno! 
        Las gotitas se abrazan fuerte cuando hace frío, como cuando tú te tapas con tu cobija. 
        ¡Así nace el hielo, como un castillo mágico de cristal! ❄️🏰✨"

        Pregunta no científica:
        "¿Cuál es tu color favorito?"

        Respuesta:
        "¡El azul, como el cielo que exploramos en el planetario del museo! 
        ¿Sabías que los colores son luz que viaja como superheroínas veloces? 
        Ven al museo y te lo muestro con un experimento divertido! 🌈⚡"

        Ahora responde esta pregunta:
        {question}
        """
        count_tokens(prompt_template)  # Contar tokens del prompt para control de costos
        print(f"Tokens del prompt: {count_tokens(prompt_template)}")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=['context', 'question']
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=True
        )
    
    def ask_question(self, question: str) -> Dict[str, str]:
        """Método principal para hacer consultas al sistema RAG"""
        try:
            response = self.qa_chain.invoke({"query": question})
            count_tokens_output = count_tokens(response['result'])
            print(f"Tokens de salida: {count_tokens_output}")
            
            return {
                "answer": response['result'],
                "sources": list(set(
                    doc.metadata.get('source', 'Desconocido') 
                    for doc in response['source_documents']
                ))
            }
        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            return {
                "answer": "¡Vaya! No pude procesar tu pregunta. Intenta reformularla.",
                "sources": []
            }

# Instancia global del sistema RAG con caché
@lru_cache(maxsize=1)
def get_rag_system():
    return RAGSystem()

# Función de interfaz simplificada
def ask_question(question: str) -> str:
    """Interfaz simplificada para otros módulos"""
    return get_rag_system().ask_question(question)["answer"]