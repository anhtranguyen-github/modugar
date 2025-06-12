import asyncio
import os
from pathlib import Path
import base64
from datetime import datetime
from rag_backend.rag_manager import RAGManager
from rag_backend.server.types import FileConfig
from rag_backend.server.helpers import LoggerManager

def read_file_as_base64(path):
    """Read a file as bytes and return a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_file_config(
    file_path: Path,
    reader_name: str,
    chunker_name: str,
    embedder_name: str,
    vector_store_name: str,
    retriever_name: str,
    generator_name: str,
) -> FileConfig:
    """Create a FileConfig with test configuration."""
    file_content = read_file_as_base64(file_path)
    
    # Set vector dimensions based on embedder
    vector_size = 768 if embedder_name == "Ollama" else (1536 if embedder_name == "OpenAI" else 384)
    
    config = {
        "Reader": {
            "selected": reader_name,
            "components": {
                "Default": {
                    "name": "Default",
                    "type": "Reader",
                    "description": "Basic text file reader",
                    "library": [],
                    "available": True,
                    "variables": [],
                    "config": {}
                }
            }
        },
        "Chunker": {
            "selected": chunker_name,
            "components": {
                "Recursive": {
                    "name": "Recursive",
                    "type": "Chunker",
                    "description": "Recursive text chunker",
                    "library": ["langchain"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Chunk Size": {
                            "type": "number",
                            "description": "Choose how many characters per chunks",
                            "values": [],
                            "value": "200"
                        },
                        "Overlap": {
                            "type": "number",
                            "description": "Choose how many characters per chunks",
                            "values": [],
                            "value": "20"
                        },
                        "Seperators": {
                            "type": "multi",
                            "description": "Select separators to split the text",
                            "values": ["\n\n", "\n", " ", ".", ",", ""],
                            "value": ""
                        }
                    }
                }
            }
        },
        "Embedder": {
            "selected": embedder_name,
            "components": {
                "OpenAI": {
                    "name": "OpenAI",
                    "type": "Embedder",
                    "description": "Test embedder using OpenAI",
                    "library": ["aiohttp"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Model": {
                            "type": "dropdown",
                            "description": "Select an OpenAI Embedding Model",
                            "values": ["text-embedding-3-small"],
                            "value": "text-embedding-3-small"
                        },
                        "vector_dimension": {
                            "type": "number",
                            "value": 1536,
                            "description": "Vector dimension for OpenAI embeddings",
                            "values": []
                        }
                    },
                    "max_batch_size": 32
                }
            }
        },
        "VectorStore": {
            "selected": vector_store_name,
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
                            "value": "test_collection"
                        },
                        "Vector Size": {
                            "type": "number",
                            "description": "Size of the vectors to be stored",
                            "values": [],
                            "value": str(vector_size)
                        },
                        "Distance": {
                            "type": "dropdown",
                            "description": "Distance metric for vector similarity",
                            "values": ["Cosine", "Euclidean", "Dot"],
                            "value": "Cosine"
                        },
                        "Host Config": {
                            "type": "dropdown",
                            "description": "Deployment configuration",
                            "values": ["local", "docker", "cloud"],
                            "value": "local"
                        }
                    }
                }
            }
        },
        "Retriever": {
            "selected": retriever_name,
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
                            "value": "Hybrid Search"
                        },
                        "Limit Mode": {
                            "type": "dropdown",
                            "description": "Method for limiting the results.",
                            "values": ["Autocut", "Fixed"],
                            "value": "Autocut"
                        },
                        "Limit/Sensitivity": {
                            "type": "number",
                            "description": "Value for limiting the results.",
                            "values": [],
                            "value": "5"
                        },
                        "Chunk Window": {
                            "type": "number",
                            "description": "Number of surrounding chunks to add to context",
                            "values": [],
                            "value": "2"
                        },
                        "Threshold": {
                            "type": "number",
                            "description": "Threshold of chunk score to apply window technique",
                            "values": [],
                            "value": "80"
                        }
                    }
                }
            }
        },
        "Generator": {
            "selected": generator_name,
            "components": {
                "Ollama": {
                    "name": "Ollama",
                    "type": "Generator",
                    "description": "Generate answers using Ollama",
                    "library": ["aiohttp"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Model": {
                            "type": "dropdown",
                            "description": "Select an Ollama model",
                            "values": ["qwen3:1.7b"],
                            "value": "qwen3:1.7b"
                        },
                        "System Message": {
                            "type": "text",
                            "description": "System message for the model",
                            "values": [],
                            "value": "You are a helpful assistant that provides accurate and detailed answers about the RAG pipeline. Use the provided context to answer questions about the implementation, components, and functionality."
                        }
                    }
                }
            }
        }
    }
    
    return FileConfig(
        filename=str(file_path),
        content=file_content,
        extension=".md",
        file_size=file_path.stat().st_size,
        labels=["fantasy", "lore", "markdown"],
        source="local",
        metadata="Fantasy world lore document",
        fileID=f"throne_of_verdant_flame",
        isURL=False,
        overwrite=True,
        rag_config=config,
        status="READY",
        status_report={}
    )

async def test_document_import():
    """Test document import functionality with Qdrant."""
    print("\n=== Testing Document Import with Qdrant ===")
    
    rag_manager = RAGManager()
    client = None
    
    try:
        # Use the existing Throne of the Verdant Flame.md file
        file_path = Path("Throne of the Verdant Flame.md")
        if not file_path.exists():
            raise Exception(f"File not found: {file_path}")
            
        file_config = create_file_config(
            file_path,
            "Default",
            "Recursive",
            "OpenAI",
            "Qdrant",
            "Advanced",
            "Ollama"
        )
        
        # 1. Connect to Qdrant
        print("1. Connecting to Qdrant...")
        client = await rag_manager.connect(
            "Qdrant",
            url="localhost",
            port=6333,
            key="",  # Empty key for local connection
            config={
                "Host Config": {"value": "local"},
                "Collection Name": {"value": "test_collection"},
                "Vector Size": {"value": "1536"},
                "Distance": {"value": "Cosine"}
            }
        )
        if not client:
            raise Exception("Failed to connect to Qdrant")
        print("✓ Successfully connected to Qdrant\n")

        # 2. Import document
        print("2. Importing document...")
        collection_name = "test_collection"
        
        # Import document
        await rag_manager.import_document(
            client,
            file_config,
            collection_name
        )
        print("✓ Document imported successfully\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\nCleaning up...")
        if client:
            try:
                await rag_manager.disconnect()
            except Exception as e:
                print(f"Warning: Error during disconnect: {str(e)}")
        print("✓ Cleanup completed")

if __name__ == "__main__":
    asyncio.run(test_document_import()) 