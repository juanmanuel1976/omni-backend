# ==============================================================================
# RAG MANAGER - Sistema RAG Real Optimizado para Crisalia v6.2
# ==============================================================================
import asyncio
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    """
    Sistema RAG real optimizado para máximo aprovechamiento del plan Render Standard.
    Usa sentence-transformers + FAISS para búsqueda semántica de alta calidad.
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Inicializa el RAG Manager con máxima potencia.
        
        Modelo all-mpnet-base-v2:
        - 384 dimensiones
        - Excelente para español e inglés
        - Balance óptimo velocidad/calidad para plan Standard
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.is_initialized = False
        
    async def initialize(self):
        """Inicialización asíncrona del modelo de embeddings."""
        if self.is_initialized:
            return
            
        try:
            logger.info(f"Inicializando modelo {self.model_name}...")
            # Inicializar en hilo separado para no bloquear FastAPI
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            self.is_initialized = True
            logger.info("RAG Manager inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando RAG Manager: {e}")
            raise
    
    def smart_text_split(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        División inteligente de texto (reemplaza RecursiveCharacterTextSplitter).
        Optimizada para máximo aprovechamiento semántico.
        """
        if len(text) <= chunk_size:
            return [text]
        
        # Separadores en orden de preferencia (párrafos > oraciones > palabras)
        separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ']
        
        chunks = []
        current_chunk = ""
        
        # División recursiva con separadores inteligentes
        def split_by_separator(text_to_split: str, separator_idx: int = 0) -> List[str]:
            if separator_idx >= len(separators):
                # Último recurso: dividir por caracteres
                return [text_to_split[i:i+chunk_size] for i in range(0, len(text_to_split), chunk_size-overlap)]
            
            separator = separators[separator_idx]
            parts = text_to_split.split(separator)
            
            result_chunks = []
            current = ""
            
            for part in parts:
                if len(current + separator + part) <= chunk_size:
                    current += (separator if current else "") + part
                else:
                    if current:
                        result_chunks.append(current)
                    
                    if len(part) > chunk_size:
                        # Dividir parte larga recursivamente
                        result_chunks.extend(split_by_separator(part, separator_idx + 1))
                        current = ""
                    else:
                        current = part
            
            if current:
                result_chunks.append(current)
                
            return result_chunks
        
        chunks = split_by_separator(text)
        
        # Aplicar overlap inteligente
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and overlap > 0:
                # Agregar overlap del chunk anterior
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                chunk = overlap_text + " " + chunk
            
            overlapped_chunks.append(chunk.strip())
        
        return overlapped_chunks
    
    async def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Crear embeddings de forma asíncrona."""
        if not self.is_initialized:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        )
        return embeddings
    
    async def index_documents(self, document_text: str, document_metadata: Dict = None) -> bool:
        """
        Indexa un documento completo para búsqueda semántica.
        Optimizado para documentos grandes en plan Standard.
        """
        try:
            # División inteligente del texto
            self.chunks = self.smart_text_split(document_text, chunk_size=1000, overlap=200)
            
            if not self.chunks:
                logger.warning("No se pudieron generar chunks del documento")
                return False
            
            # Generar embeddings
            logger.info(f"Generando embeddings para {len(self.chunks)} chunks...")
            embeddings = await self.create_embeddings(self.chunks)
            
            # Crear índice FAISS optimizado
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product para cosine similarity
            
            # Normalizar embeddings para cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Agregar al índice
            self.index.add(embeddings.astype(np.float32))
            
            # Guardar metadata de chunks
            self.chunk_metadata = [
                {
                    "chunk_id": i,
                    "text": chunk,
                    "document_metadata": document_metadata or {},
                    "chunk_size": len(chunk)
                }
                for i, chunk in enumerate(self.chunks)
            ]
            
            logger.info(f"Documento indexado: {len(self.chunks)} chunks, {dimension} dimensiones")
            return True
            
        except Exception as e:
            logger.error(f"Error indexando documento: {e}")
            return False
    
    async def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Búsqueda semántica de alta calidad.
        Retorna los chunks más relevantes para la query.
        """
        if not self.index or not self.is_initialized:
            logger.warning("RAG Manager no inicializado o sin documentos indexados")
            return []
        
        try:
            # Generar embedding de la query
            query_embedding = await self.create_embeddings([query])
            
            # Normalizar para cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Búsqueda en el índice
            scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
            
            # Preparar resultados
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result["relevance_score"] = float(score)
                    results.append(result)
            
            # Filtrar resultados con score mínimo
            results = [r for r in results if r["relevance_score"] > 0.3]
            
            logger.info(f"Búsqueda semántica: {len(results)} chunks relevantes encontrados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {e}")
            return []
    
    async def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """
        Obtiene el contexto más relevante para una query específica.
        Optimizado para no exceder límites de tokens.
        """
        # Búsqueda semántica
        relevant_chunks = await self.semantic_search(query, top_k=8)
        
        if not relevant_chunks:
            return ""
        
        # Construcción de contexto optimizada
        context_parts = []
        current_length = 0
        
        for chunk_data in relevant_chunks:
            chunk_text = chunk_data["text"]
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length <= max_context_length:
                # Agregar con metadata de relevancia
                relevance = chunk_data["relevance_score"]
                context_parts.append(f"[Relevancia: {relevance:.2f}]\n{chunk_text}")
                current_length += chunk_length + 50  # Buffer para metadata
            else:
                # Truncar chunk si es necesario
                remaining_space = max_context_length - current_length
                if remaining_space > 200:  # Solo si vale la pena
                    truncated = chunk_text[:remaining_space-50]
                    context_parts.append(f"[Relevancia: {relevance:.2f}]\n{truncated}...")
                break
        
        context = "\n\n---\n\n".join(context_parts)
        
        logger.info(f"Contexto generado: {len(context)} caracteres de {len(relevant_chunks)} chunks")
        return context
    
    def get_stats(self) -> Dict:
        """Estadísticas del RAG Manager para debugging."""
        return {
            "is_initialized": self.is_initialized,
            "model_name": self.model_name,
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.model.get_sentence_embedding_dimension() if self.model else 0
        }
    
    def clear_index(self):
        """Limpiar índice y liberar memoria."""
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        logger.info("Índice RAG limpiado")

# Instancia global del RAG Manager
rag_manager = RAGManager()
