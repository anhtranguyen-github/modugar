from wasabi import msg

import weaviate
from weaviate.client import WeaviateAsyncClient
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter, Sort, MetadataQuery
from weaviate.collections.classes.data import DataObject
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.init import AdditionalConfig, Timeout

import os
import asyncio
import json
import re
from urllib.parse import urlparse
from datetime import datetime
from typing import Dict, Any, Optional, List

from sklearn.decomposition import PCA

from rag_backend.components.document import Document
from rag_backend.components.interfaces import (
    Reader,
    Chunker,
    Embedding,
    Retriever,
    Generator,
    VectorStore,
)
from rag_backend.server.helpers import LoggerManager
from rag_backend.server.types import FileConfig, FileStatus

# Import Readers
from rag_backend.components.reader.BasicReader import BasicReader
from rag_backend.components.reader.HTMLReader import HTMLReader
from rag_backend.components.reader.DoclingReader import DoclingReader

# Import Chunkers
from rag_backend.components.chunking.TokenChunker import TokenChunker
from rag_backend.components.chunking.SentenceChunker import SentenceChunker
from rag_backend.components.chunking.RecursiveChunker import RecursiveChunker
from rag_backend.components.chunking.HTMLChunker import HTMLChunker
from rag_backend.components.chunking.MarkdownChunker import MarkdownChunker
from rag_backend.components.chunking.CodeChunker import CodeChunker
from rag_backend.components.chunking.JSONChunker import JSONChunker
from rag_backend.components.chunking.SemanticChunker import SemanticChunker

# Import Embedders
from rag_backend.components.embedding.OllamaEmbedder import OllamaEmbedder
from rag_backend.components.embedding.SentenceTransformersEmbedder import (
    SentenceTransformersEmbedder,
)
from rag_backend.components.embedding.OpenAIEmbedder import OpenAIEmbedder

# Import Retrievers
from rag_backend.components.retriever.WindowRetriever import WindowRetriever

# Import Generators
from rag_backend.components.generation.OllamaGenerator import OllamaGenerator
from rag_backend.components.generation.OpenAIGenerator import OpenAIGenerator

# Import Vector Stores
from rag_backend.components.vector_store.WeaviateStore import WeaviateStore
from rag_backend.components.vector_store.QdrantStore import QdrantStore

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    msg.warn("tiktoken not installed, token counting will be disabled.")
    TIKTOKEN_AVAILABLE = False

### Add new components here ###

readers = [
    BasicReader(),
    HTMLReader(),
    DoclingReader(),
]

chunkers = [
    TokenChunker(),
    SentenceChunker(),
    RecursiveChunker(),
    SemanticChunker(),
    HTMLChunker(),
    MarkdownChunker(),
    CodeChunker(),
    JSONChunker(),
]

embedders = [
    OllamaEmbedder(),
    SentenceTransformersEmbedder(),
    OpenAIEmbedder(),
]

retrievers = [WindowRetriever()]

generators = [OllamaGenerator(), OpenAIGenerator()]

vector_stores = [
    WeaviateStore(),
    QdrantStore(),
]

### ----------------------- ###


class ReaderManager:
    def __init__(self):
        self.readers: dict[str, Reader] = {reader.name: reader for reader in readers}

    async def load(
        self, reader: str, fileConfig: FileConfig, logger: LoggerManager
    ) -> list[Document]:
        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            if reader in self.readers:
                config = fileConfig.rag_config["Reader"].components[reader].config
                documents: list[Document] = await self.readers[reader].load(
                    config, fileConfig
                )
                for document in documents:
                    document.meta["Reader"] = (
                        fileConfig.rag_config["Reader"].components[reader].model_dump()
                    )
                elapsed_time = round(loop.time() - start_time, 2)
                if len(documents) == 1:
                    await logger.send_report(
                        fileConfig.fileID,
                        FileStatus.LOADING,
                        f"Loaded {fileConfig.filename}",
                        took=elapsed_time,
                    )
                else:
                    await logger.send_report(
                        fileConfig.fileID,
                        FileStatus.LOADING,
                        f"Loaded {fileConfig.filename} with {len(documents)} documents",
                        took=elapsed_time,
                    )
                await logger.send_report(
                    fileConfig.fileID, FileStatus.CHUNKING, "", took=0
                )
                return documents
            else:
                raise Exception(f"{reader} Reader not found")

        except Exception as e:
            raise Exception(f"Reader {reader} failed with: {str(e)}")


class ChunkerManager:
    def __init__(self):
        self.chunkers: dict[str, Chunker] = {
            chunker.name: chunker for chunker in chunkers
        }

    async def chunk(
        self,
        chunker: str,
        fileConfig: FileConfig,
        documents: list[Document],
        embedder: Embedding,
        logger: LoggerManager,
    ) -> list[Document]:
        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            if chunker in self.chunkers:
                config = fileConfig.rag_config["Chunker"].components[chunker].config
                embedder_config = (
                    fileConfig.rag_config["Embedder"].components[embedder.name].config
                )
                chunked_documents = await self.chunkers[chunker].chunk(
                    config=config,
                    documents=documents,
                    embedder=embedder,
                    embedder_config=embedder_config,
                )
                for chunked_document in chunked_documents:
                    chunked_document.meta["Chunker"] = (
                        fileConfig.rag_config["Chunker"]
                        .components[chunker]
                        .model_dump()
                    )
                elapsed_time = round(loop.time() - start_time, 2)
                if len(documents) == 1:
                    await logger.send_report(
                        fileConfig.fileID,
                        FileStatus.CHUNKING,
                        f"Split {fileConfig.filename} into {len(chunked_documents[0].chunks)} chunks",
                        took=elapsed_time,
                    )
                else:
                    await logger.send_report(
                        fileConfig.fileID,
                        FileStatus.CHUNKING,
                        f"Chunked all {len(chunked_documents)} documents with a total of {sum([len(document.chunks) for document in chunked_documents])} chunks",
                        took=elapsed_time,
                    )

                await logger.send_report(
                    fileConfig.fileID, FileStatus.EMBEDDING, "", took=0
                )
                return chunked_documents
            else:
                raise Exception(f"{chunker} Chunker not found")
        except Exception as e:
            raise e


class EmbeddingManager:
    def __init__(self):
        self.embedders: dict[str, Embedding] = {
            embedder.name: embedder for embedder in embedders
        }

    async def vectorize(
        self,
        embedder: str,
        fileConfig: FileConfig,
        documents: list[Document],
        logger: LoggerManager,
    ) -> list[Document]:
        """Vectorizes chunks in batches
        @parameter: documents : Document - Verba document
        @returns Document - Document with vectorized chunks
        """
        try:
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            if embedder in self.embedders:
                config = fileConfig.rag_config["Embedder"].components[embedder].config

                for document in documents:
                    content = [
                        document.metadata + "\n" + chunk.content
                        for chunk in document.chunks
                    ]
                    embeddings = await self.batch_vectorize(embedder, config, content)

                    if len(embeddings) >= 3:
                        pca = PCA(n_components=3)
                        generated_pca_embeddings = pca.fit_transform(embeddings)
                        pca_embeddings = [
                            pca_.tolist() for pca_ in generated_pca_embeddings
                        ]
                    else:
                        pca_embeddings = [embedding[0:3] for embedding in embeddings]

                    for vector, chunk, pca_ in zip(
                        embeddings, document.chunks, pca_embeddings
                    ):
                        chunk.vector = vector
                        chunk.pca = pca_

                    document.meta["Embedder"] = (
                        fileConfig.rag_config["Embedder"]
                        .components[embedder]
                        .model_dump()
                    )

                elapsed_time = round(loop.time() - start_time, 2)
                await logger.send_report(
                    fileConfig.fileID,
                    FileStatus.EMBEDDING,
                    f"Vectorized all chunks",
                    took=elapsed_time,
                )
                await logger.send_report(
                    fileConfig.fileID, FileStatus.INGESTING, "", took=0
                )
                return documents
            else:
                raise Exception(f"{embedder} Embedder not found")
        except Exception as e:
            raise e

    async def batch_vectorize(
        self, embedder: str, config: dict, content: list[str]
    ) -> list[list[float]]:
        """Vectorize content in batches"""
        try:
            batches = [
                content[i : i + self.embedders[embedder].max_batch_size]
                for i in range(0, len(content), self.embedders[embedder].max_batch_size)
            ]
            msg.info(f"Vectorizing {len(content)} chunks in {len(batches)} batches")
            tasks = [
                self.embedders[embedder].vectorize(config, batch) for batch in batches
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check if all tasks were successful
            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                error_messages = [str(e) for e in errors]
                raise Exception(
                    f"Vectorization failed for some batches: {', '.join(error_messages)}"
                )

            # Flatten the results
            flattened_results = [item for sublist in results for item in sublist]

            # Verify the number of vectors matches the input content
            if len(flattened_results) != len(content):
                raise Exception(
                    f"Mismatch in vectorization results: expected {len(content)} vectors, got {len(flattened_results)}"
                )

            return flattened_results
        except Exception as e:
            raise Exception(f"Batch vectorization failed: {str(e)}")

    async def vectorize_query(
        self, embedder: str, content: str, rag_config: dict
    ) -> list[float]:
        try:
            if embedder in self.embedders:
                config = rag_config["Embedder"].components[embedder].config
                embeddings = await self.embedders[embedder].vectorize(config, [content])
                return embeddings[0]
            else:
                raise Exception(f"{embedder} Embedder not found")
        except Exception as e:
            raise e


class RetrieverManager:
    def __init__(self):
        self.retrievers: dict[str, Retriever] = {
            retriever.name: retriever for retriever in retrievers
        }
        self.vector_store_manager = VectorStoreManager()

    async def retrieve(
        self,
        client,
        retriever: str,
        query: str,
        vector: list[float],
        rag_config: dict,
        labels: list[str],
        document_uuids: list[str],
    ):
        try:
            if retriever not in self.retrievers:
                raise Exception(f"Retriever {retriever} not found")

            embedder_model = (
                rag_config["Embedder"]
                .components[rag_config["Embedder"].selected]
                .config["Model"]
                .value
            )
            config = rag_config["Retriever"].components[retriever].config

            # Set vector store manager in retriever
            retriever_instance = self.retrievers[retriever]
            retriever_instance._vector_store_manager = self.vector_store_manager

            documents, context = await retriever_instance.retrieve(
                client,
                query,
                vector,
                config,
                embedder_model,
                labels,
                document_uuids,
            )
            return (documents, context)

        except Exception as e:
            raise e


class GeneratorManager:
    def __init__(self):
        self.generators: dict[str, Generator] = {
            generator.name: generator for generator in generators
        }

    async def generate_stream(self, rag_config, query, context, conversation):
        """Generate a stream of response dicts based on a list of queries and list of contexts, and includes conversational context
        @parameter: queries : list[str] - List of queries
        @parameter: context : list[str] - List of contexts
        @parameter: conversation : dict - Conversational context
        @returns Iterator[dict] - Token response generated by the Generator in this format {system:TOKEN, finish_reason:stop or empty}.
        """

        generator = rag_config["Generator"].selected
        generator_config = (
            rag_config["Generator"].components[rag_config["Generator"].selected].config
        )

        if generator not in self.generators:
            raise Exception(f"Generator {generator} not found")

        async for result in self.generators[generator].generate_stream(
            generator_config, query, context, conversation
        ):
            yield result

    def truncate_conversation_dicts(
        self, conversation_dicts: list[dict[str, any]], max_tokens: int
    ) -> list[dict[str, any]]:
        """
        Truncate a list of conversation dictionaries to fit within a specified maximum token limit.

        @parameter conversation_dicts: List[Dict[str, any]] - A list of conversation dictionaries that may contain various keys, where 'content' key is present and contains text data.
        @parameter max_tokens: int - The maximum number of tokens that the combined content of the truncated conversation dictionaries should not exceed.

        @returns List[Dict[str, any]]: A list of conversation dictionaries that have been truncated so that their combined content respects the max_tokens limit. The list is returned in the original order of conversation with the most recent conversation being truncated last if necessary.
        """
        if not TIKTOKEN_AVAILABLE:
            # If tiktoken is not available, return the original conversation without truncation
            return conversation_dicts

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        accumulated_tokens = 0
        truncated_conversation_dicts = []

        # Start with the newest conversations
        for item_dict in reversed(conversation_dicts):
            item_tokens = encoding.encode(item_dict["content"], disallowed_special=())

            # If adding the entire new item exceeds the max tokens
            if accumulated_tokens + len(item_tokens) > max_tokens:
                # Calculate how many tokens we can add from this item
                remaining_space = max_tokens - accumulated_tokens
                truncated_content = encoding.decode(item_tokens[:remaining_space])

                # Create a new truncated item dictionary
                truncated_item_dict = {
                    "type": item_dict["type"],
                    "content": truncated_content,
                    "typewriter": item_dict["typewriter"],
                }

                truncated_conversation_dicts.append(truncated_item_dict)
                break

            truncated_conversation_dicts.append(item_dict)
            accumulated_tokens += len(item_tokens)

        # The list has been built in reverse order so we reverse it again
        return list(reversed(truncated_conversation_dicts))


class VectorStoreManager:
    def __init__(self):
        self.stores: Dict[str, VectorStore] = {
            store.name: store for store in vector_stores
        }

    async def connect(self, store_name: str, **kwargs) -> Any:
        """Connect to a vector store
        @parameter: store_name : str - Name of the vector store to connect to
        @parameter: **kwargs - Additional connection parameters
        @returns Any - Vector store client instance
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")

            client = await self.stores[store_name].connect(**kwargs)
            if client:
                return client
            return None
        except Exception as e:
            raise Exception(f"Failed to connect to vector store {store_name}: {str(e)}")

    async def disconnect(self, store_name: str, client: Any) -> bool:
        """Disconnect from a vector store
        @parameter: store_name : str - Name of the vector store to disconnect from
        @parameter: client : Any - Vector store client instance
        @returns bool - True if disconnected successfully
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].disconnect(client)
        except Exception as e:
            raise Exception(
                f"Failed to disconnect from vector store {store_name}: {str(e)}"
            )

    async def is_ready(self, store_name: str, client: Any) -> bool:
        """Check if a vector store is ready
        @parameter: store_name : str - Name of the vector store to check
        @parameter: client : Any - Vector store client instance
        @returns bool - True if store is ready
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].is_ready(client)
        except Exception as e:
            raise Exception(
                f"Failed to check vector store {store_name} readiness: {str(e)}"
            )

    async def create_collection(
        self, store_name: str, client: Any, collection_name: str, **kwargs
    ) -> bool:
        """Create a new collection in a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to create
        @parameter: **kwargs - Additional collection creation parameters
        @returns bool - True if collection created successfully
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].create_collection(
                client, collection_name, **kwargs
            )
        except Exception as e:
            raise Exception(
                f"Failed to create collection in vector store {store_name}: {str(e)}"
            )

    async def delete_collection(
        self, store_name: str, client: Any, collection_name: str
    ) -> bool:
        """Delete a collection from a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to delete
        @returns bool - True if collection deleted successfully
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].delete_collection(
                client, collection_name
            )
        except Exception as e:
            raise Exception(
                f"Failed to delete collection from vector store {store_name}: {str(e)}"
            )

    async def insert_vectors(
        self,
        store_name: str,
        client: Any,
        collection_name: str,
        vectors: list[list[float]],
        metadata: list[dict],
        **kwargs,
    ) -> list[str]:
        """Insert vectors into a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to insert into
        @parameter: vectors : list[list[float]] - List of vectors to insert
        @parameter: metadata : list[dict] - List of metadata for each vector
        @parameter: **kwargs - Additional insertion parameters
        @returns list[str] - List of inserted vector IDs
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].insert_vectors(
                client, collection_name, vectors, metadata, **kwargs
            )
        except Exception as e:
            raise Exception(
                f"Failed to insert vectors into vector store {store_name}: {str(e)}"
            )

    async def search_vectors(
        self,
        store_name: str,
        client: Any,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        **kwargs,
    ) -> list[dict]:
        """Search for similar vectors in a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to search in
        @parameter: query_vector : list[float] - Query vector to search with
        @parameter: limit : int - Maximum number of results to return
        @parameter: **kwargs - Additional search parameters
        @returns list[dict] - List of search results
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].search_vectors(
                client, collection_name, query_vector, limit, **kwargs
            )
        except Exception as e:
            raise Exception(
                f"Failed to search vectors in vector store {store_name}: {str(e)}"
            )

    async def delete_vectors(
        self, store_name: str, client: Any, collection_name: str, vector_ids: list[str]
    ) -> bool:
        """Delete vectors from a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to delete from
        @parameter: vector_ids : list[str] - List of vector IDs to delete
        @returns bool - True if vectors deleted successfully
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].delete_vectors(
                client, collection_name, vector_ids
            )
        except Exception as e:
            raise Exception(
                f"Failed to delete vectors from vector store {store_name}: {str(e)}"
            )

    async def get_collection_info(
        self, store_name: str, client: Any, collection_name: str
    ) -> dict:
        """Get information about a collection in a vector store
        @parameter: store_name : str - Name of the vector store
        @parameter: client : Any - Vector store client instance
        @parameter: collection_name : str - Name of collection to get info for
        @returns dict - Collection information
        """
        try:
            if store_name not in self.stores:
                raise Exception(f"Vector store {store_name} not found")
            return await self.stores[store_name].get_collection_info(
                client, collection_name
            )
        except Exception as e:
            raise Exception(
                f"Failed to get collection info from vector store {store_name}: {str(e)}"
            )
