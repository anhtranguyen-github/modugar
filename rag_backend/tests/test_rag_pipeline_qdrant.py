import asyncio
import os
from pathlib import Path
import base64
from rag_backend.components.managers import (
    ReaderManager,
    ChunkerManager,
    EmbeddingManager,
    RetrieverManager,
    VectorStoreManager,
)
from rag_backend.server.types import FileConfig
from rag_backend.server.helpers import LoggerManager
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message=".*validate_tree.*", category=DeprecationWarning
)
# Suppress NumPy warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)


def read_file_as_base64(path):
    """Read a file as bytes and return a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_test_pdf(path):
    """Create a test PDF file with sample content."""
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(100, 750, "Test Document")
    c.drawString(100, 730, "This is a test document for the RAG pipeline with Qdrant.")
    c.drawString(100, 710, "Section 1: Introduction")
    c.drawString(100, 690, "- Point 1: Document Reading")
    c.drawString(100, 670, "- Point 2: Text Chunking")
    c.drawString(100, 650, "- Point 3: Vector Embedding")
    c.drawString(100, 610, "Section 2: Technical Details")
    c.drawString(100, 590, "Here we have some technical details about the implementation.")
    c.drawString(100, 570, "The RAG pipeline consists of several key components:")
    c.drawString(100, 550, "1. Document Reader: Processes input files and extracts text")
    c.drawString(100, 530, "2. Chunker: Splits text into manageable chunks")
    c.drawString(100, 510, "3. Embedder: Converts text chunks into vector embeddings")
    c.save()


async def test_rag_pipeline_qdrant():
    """Test the RAG pipeline with Qdrant vector store"""
    # Initialize managers
    reader_manager = ReaderManager()
    chunker_manager = ChunkerManager()
    embedder_manager = EmbeddingManager()
    retriever_manager = RetrieverManager()
    vector_store_manager = VectorStoreManager()
    logger = LoggerManager()

    # Create test directory if it doesn't exist
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a test PDF file
    test_file = test_dir / "test_document.pdf"
    create_test_pdf(test_file)

    # Read the file content and encode as base64
    file_content = read_file_as_base64(test_file)

    file_config = FileConfig(
        filename=str(test_file),
        content=file_content,  # Pass base64 encoded content
        extension=".pdf",  # Updated to PDF
        file_size=len(file_content),
        labels=["test", "pdf"],
        source="local",
        metadata="Test document for RAG pipeline with Qdrant",
        fileID="test_001",
        isURL=False,
        overwrite=True,
        rag_config={
            "Reader": {
                "selected": "Docling",
                "components": {
                    "Docling": {
                        "name": "Docling",
                        "type": "Reader",
                        "description": "Test reader",
                        "library": ["docling"],
                        "available": True,
                        "variables": [],
                        "config": {
                            "use_models": {
                                "type": "boolean",
                                "description": "Whether to use Docling models for processing",
                                "values": ["true", "false"],
                                "value": "false",
                            }
                        },
                    }
                },
            },
            "Chunker": {
                "selected": "Recursive",
                "components": {
                    "Recursive": {
                        "name": "Recursive",
                        "type": "Chunker",
                        "description": "Test chunker",
                        "library": ["langchain"],
                        "available": True,
                        "variables": [],
                        "config": {
                            "Chunk Size": {
                                "type": "number",
                                "description": "Choose how many characters per chunks",
                                "values": [],
                                "value": "200",
                            },
                            "Overlap": {
                                "type": "number",
                                "description": "Choose how many characters per chunks",
                                "values": [],
                                "value": "20",
                            },
                            "Seperators": {
                                "type": "multi",
                                "description": "Select separators to split the text",
                                "values": [
                                    "\n\n",
                                    "\n",
                                    " ",
                                    ".",
                                    ",",
                                    "\u200b",
                                    "\uff0c",
                                    "\u3001",
                                    "\uff0e",
                                    "\u3002",
                                    "",
                                ],
                                "value": "",
                            },
                        },
                    }
                },
            },
            "Embedder": {
                "selected": "SentenceTransformers",
                "components": {
                    "SentenceTransformers": {
                        "name": "SentenceTransformers",
                        "type": "Embedder",
                        "description": "Test embedder using sentence-transformers",
                        "library": ["sentence_transformers"],
                        "available": True,
                        "variables": [],
                        "config": {
                            "Model": {
                                "type": "dropdown",
                                "description": "Select an HuggingFace Embedding Model",
                                "values": [
                                    "all-MiniLM-L6-v2",
                                    "mixedbread-ai/mxbai-embed-large-v1",
                                    "all-mpnet-base-v2",
                                    "BAAI/bge-m3",
                                    "all-MiniLM-L12-v2",
                                    "paraphrase-MiniLM-L6-v2",
                                ],
                                "value": "all-MiniLM-L6-v2",
                            }
                        },
                        "max_batch_size": 32,
                    }
                },
            },
            "VectorStore": {
                "selected": "Qdrant",
                "components": {
                    "Qdrant": {
                        "name": "Qdrant",
                        "type": "VectorStore",
                        "description": "Qdrant vector store for storing and retrieving embeddings",
                        "library": ["qdrant_client"],
                        "available": True,
                        "variables": [],
                        "config": {
                            "Collection Name": {
                                "type": "text",
                                "description": "Name of the collection to store vectors",
                                "values": [],
                                "value": "test_collection",
                            },
                            "Vector Size": {
                                "type": "number",
                                "description": "Size of the vectors to be stored",
                                "values": [],
                                "value": "384",
                            },
                        },
                    }
                },
            },
            "Retriever": {
                "selected": "Advanced",
                "components": {
                    "Advanced": {
                        "name": "Advanced",
                        "type": "Retriever",
                        "description": "Advanced retriever with window-based context",
                        "library": [],
                        "available": True,
                        "variables": [],
                        "config": {
                            "Search Mode": {
                                "type": "dropdown",
                                "description": "Switch between search types.",
                                "values": ["Hybrid Search"],
                                "value": "Hybrid Search",
                            },
                            "Limit Mode": {
                                "type": "dropdown",
                                "description": "Method for limiting the results.",
                                "values": ["Autocut", "Fixed"],
                                "value": "Autocut",
                            },
                            "Limit/Sensitivity": {
                                "type": "number",
                                "description": "Value for limiting the results.",
                                "values": [],
                                "value": "5",
                            },
                            "Chunk Window": {
                                "type": "number",
                                "description": "Number of surrounding chunks to add to context",
                                "values": [],
                                "value": "2",
                            },
                            "Threshold": {
                                "type": "number",
                                "description": "Threshold of chunk score to apply window technique",
                                "values": [],
                                "value": "80",
                            },
                        },
                    }
                },
            },
        },
        status="READY",
        status_report={},
    )

    try:
        # Step 1: Read the document
        print("\n1. Reading document...")
        documents = await reader_manager.load(
            file_config.rag_config["Reader"].selected, file_config, logger
        )
        print(f"ReaderManager returned {len(documents)} documents")

        # Step 2: Chunk the documents
        print("\n2. Chunking documents...")
        chunked_documents = await chunker_manager.chunk(
            file_config.rag_config["Chunker"].selected,
            file_config,
            documents,
            embedder_manager.embedders[file_config.rag_config["Embedder"].selected],
            logger,
        )
        print(f"Created {sum(len(doc.chunks) for doc in chunked_documents)} chunks")

        # Step 3: Embed the chunks
        print("\n3. Embedding chunks...")
        vectorized_documents = await embedder_manager.vectorize(
            file_config.rag_config["Embedder"].selected,
            file_config,
            chunked_documents,
            logger,
        )
        print(f"Vectorized {len(vectorized_documents)} documents")

        # Step 4: Store in Qdrant
        print("\n4. Storing in Qdrant...")

        # Initialize Qdrant connection
        client = await vector_store_manager.initialize_store(
            "Qdrant", url="localhost", port=6333
        )
        if not client:
            raise Exception("Failed to connect to Qdrant")

        # Get vector store config
        vector_store_config = (
            file_config.rag_config["VectorStore"].components["Qdrant"].config
        )
        collection_name = vector_store_config["Collection Name"].value
        vector_size = int(vector_store_config["Vector Size"].value)

        # Create collection if it doesn't exist
        await vector_store_manager.create_collection(
            collection_name, vector_size=vector_size
        )

        # Prepare vectors and metadata for insertion
        vectors = []
        metadata = []
        for doc in vectorized_documents:
            for chunk in doc.chunks:
                vectors.append(chunk.vector)
                metadata.append(
                    {
                        "text": chunk.content,
                        "doc_id": file_config.fileID,
                        "chunk_id": chunk.chunk_id,
                    }
                )

        # Insert vectors
        vector_ids = await vector_store_manager.insert_vectors(
            collection_name, vectors, metadata
        )
        print(f"Successfully stored {len(vector_ids)} vectors in Qdrant")

        # Step 5: Test retrieval
        print("\n5. Testing retrieval...")
        query = "What are the key components of the RAG pipeline?"

        # Get query vector using the embedder
        query_vector = await embedder_manager.vectorize_query(
            file_config.rag_config["Embedder"].selected, query, file_config.rag_config
        )

        # Get the retriever and set its vector store manager
        retriever = retriever_manager.retrievers[
            file_config.rag_config["Retriever"].selected
        ]
        retriever._vector_store_manager = vector_store_manager

        # Use retriever to get relevant documents and context
        documents, context = await retriever.retrieve(
            client=client,
            query=query,
            vector=query_vector,
            config=file_config.rag_config["Retriever"]
            .components[file_config.rag_config["Retriever"].selected]
            .config,
            embedder=file_config.rag_config["Embedder"].selected,
            labels=file_config.labels,
            document_uuids=[file_config.fileID],
            collection_name=collection_name
        )

        print(f"\nRetrieved {len(documents)} documents")
        print("\nContext:")
        print(context)

        # Clean up
        await vector_store_manager.disconnect()

    except Exception as e:
        print(f"\nError in test: {str(e)}")
        import traceback

        print("\nFull error traceback:")
        print(traceback.format_exc())
        # Re-raise the exception to ensure the test fails
        raise
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    try:
        print("Starting RAG pipeline test...")
        asyncio.run(test_rag_pipeline_qdrant())
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback

        print("\nFull error traceback:")
        print(traceback.format_exc())
        exit(1)