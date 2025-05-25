import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from rag_backend.components.document import Document
from rag_backend.server.types import FileConfig, FileStatus, ChunkScore, Credentials
from rag_backend.server.helpers import LoggerManager
from rag_backend.components.managers import (
    ReaderManager,
    ChunkerManager,
    EmbeddingManager,
    RetrieverManager,
    GeneratorManager,
    VectorStoreManager
)

class RAGManager:
    """Manages RAG pipeline components with VectorStoreManager."""

    def __init__(self) -> None:
        self.reader_manager = ReaderManager()
        self.chunker_manager = ChunkerManager()
        self.embedder_manager = EmbeddingManager()
        self.retriever_manager = RetrieverManager()
        self.generator_manager = GeneratorManager()
        self.vector_store_manager = VectorStoreManager()
        self.logger = LoggerManager()

    async def connect(self, store_name: str, **kwargs) -> Any:
        """Connect to the vector store."""
        try:
            client = await self.vector_store_manager.initialize_store(store_name, **kwargs)
            if client:
                return client
            raise Exception(f"Failed to connect to {store_name}")
        except Exception as e:
            raise Exception(f"Connection failed: {str(e)}")

    async def disconnect(self) -> bool:
        """Disconnect from the vector store."""
        return await self.vector_store_manager.disconnect()

    async def import_document(
        self,
        client: Any,
        file_config: FileConfig,
        collection_name: str,
        logger: Optional[LoggerManager] = None
    ) -> None:
        """Import a document into the vector store."""
        if logger is None:
            logger = self.logger

        try:
            start_time = asyncio.get_event_loop().time()

            # Create collection if it doesn't exist
            collection_created = await self.vector_store_manager.create_collection(
                collection_name,
                vector_size=384  # Default size for sentence-transformers
            )
            if not collection_created:
                raise Exception(f"Failed to create collection {collection_name}")

            # Read document
            documents = await self.reader_manager.load(
                file_config.rag_config["Reader"].selected,
                file_config,
                logger
            )

            # Process each document
            for document in documents:
                # Chunk document
                chunked_documents = await self.chunker_manager.chunk(
                    file_config.rag_config["Chunker"].selected,
                    file_config,
                    [document],
                    self.embedder_manager.embedders[file_config.rag_config["Embedder"].selected],
                    logger
                )

                # Embed chunks
                vectorized_documents = await self.embedder_manager.vectorize(
                    file_config.rag_config["Embedder"].selected,
                    file_config,
                    chunked_documents,
                    logger
                )

                # Store in vector store
                for doc in vectorized_documents:
                    vectors = []
                    metadata = []
                    for chunk in doc.chunks:
                        vectors.append(chunk.vector)
                        metadata.append({
                            "text": chunk.content,
                            "doc_id": file_config.fileID,
                            "chunk_id": chunk.chunk_id,
                            "title": doc.title,
                            "labels": doc.labels
                        })

                    vector_ids = await self.vector_store_manager.insert_vectors(
                        collection_name,
                        vectors,
                        metadata
                    )
                    if not vector_ids:
                        raise Exception(f"Failed to insert vectors into collection {collection_name}")

            elapsed_time = round(asyncio.get_event_loop().time() - start_time, 2)
            await logger.send_report(
                file_config.fileID,
                FileStatus.DONE,
                f"Successfully imported {file_config.filename}",
                took=elapsed_time
            )

        except Exception as e:
            await logger.send_report(
                file_config.fileID,
                FileStatus.ERROR,
                f"Import failed: {str(e)}",
                took=0
            )
            raise

    async def retrieve_chunks(
        self,
        client: Any,
        query: str,
        rag_config: dict,
        collection_name: str,
        labels: List[str] = None,
        document_uuids: List[str] = None
    ) -> tuple[List[dict], str]:
        """Retrieve relevant chunks for a query."""
        try:
            # Get query vector
            embedder = rag_config["Embedder"].selected
            query_vector = await self.embedder_manager.vectorize_query(
                embedder,
                query,
                rag_config
            )

            # Get retriever and set vector store manager
            retriever = self.retriever_manager.retrievers[rag_config["Retriever"].selected]
            retriever._vector_store_manager = self.vector_store_manager

            # Retrieve documents and context
            documents, context = await retriever.retrieve(
                client=client,
                query=query,
                vector=query_vector,
                config=rag_config["Retriever"].components[rag_config["Retriever"].selected].config,
                embedder=embedder,
                labels=labels or [],
                document_uuids=document_uuids or []
            )

            return documents, context

        except Exception as e:
            raise Exception(f"Retrieval failed: {str(e)}")

    async def generate_answer(
        self,
        rag_config: dict,
        query: str,
        context: str,
        conversation: List[dict] = None
    ):
        """Generate an answer using the context and conversation history."""
        try:
            async for result in self.generator_manager.generate_stream(
                rag_config,
                query,
                context,
                conversation or []
            ):
                yield result
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")

    async def get_document_content(
        self,
        client: Any,
        collection_name: str,
        doc_id: str,
        page: int = 1,
        chunks_per_page: int = 10
    ) -> tuple[List[dict], int]:
        """Get document content with pagination."""
        try:
            # Get document info
            doc_info = await self.vector_store_manager.get_collection_info(collection_name)
            if not doc_info:
                raise Exception("Document not found")

            # Calculate chunk range
            start_idx = (page - 1) * chunks_per_page
            end_idx = start_idx + chunks_per_page

            # Search for chunks
            results = await self.vector_store_manager.search_vectors(
                collection_name,
                query_vector=[0] * 384,  # Dummy vector for getting all chunks
                limit=chunks_per_page,
                filters={
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        },
                        {
                            "key": "chunk_id",
                            "range": {
                                "gte": start_idx,
                                "lt": end_idx
                            }
                        }
                    ]
                }
            )

            # Format results
            content_pieces = []
            for result in results:
                content_pieces.append({
                    "content": result["metadata"]["text"],
                    "chunk_id": result["metadata"]["chunk_id"],
                    "score": result["score"],
                    "type": "text"
                })

            # Calculate total pages
            total_chunks = len(results)
            total_pages = (total_chunks + chunks_per_page - 1) // chunks_per_page

            return content_pieces, total_pages

        except Exception as e:
            raise Exception(f"Failed to get document content: {str(e)}") 