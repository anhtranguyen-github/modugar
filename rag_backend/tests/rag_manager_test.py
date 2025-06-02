import asyncio
import os
from pathlib import Path
import base64
from rag_backend.rag_manager import RAGManager
from rag_backend.server.types import FileConfig
from rag_backend.server.helpers import LoggerManager
from rag_backend.components.types import InputConfig

def read_file_as_base64(path):
    """Read a file as bytes and return a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_test_file():
    """Create a test file with sample content."""
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "test_document.md"
    test_content = """# Test Document

This is a test document for the RAG pipeline.

## Section 1: Introduction
This is the first section with some important information about the RAG pipeline.
- Point 1: Document Reading
- Point 2: Text Chunking
- Point 3: Vector Embedding

## Section 2: Technical Details
Here we have some technical details about the implementation.
The RAG pipeline consists of several key components:

1. Document Reader: Processes input files and extracts text
2. Chunker: Splits text into manageable chunks
3. Embedder: Converts text chunks into vector embeddings
"""
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    return test_file

def create_file_config(test_file, reader="Default", chunker="Recursive", embedder="SentenceTransformers"):
    """Create a FileConfig with test configuration."""
    file_content = read_file_as_base64(test_file)
    
    config = {
        "Reader": {
            "selected": reader,
            "components": {
                "Default": {
                    "name": "Default",
                    "type": "Reader",
                    "description": "Basic text file reader",
                    "library": [],
                    "available": True,
                    "variables": [],
                    "config": {}
                },
                "Docling": {
                    "name": "Docling",
                    "type": "Reader",
                    "description": "Advanced document reader",
                    "library": ["docling"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "use_models": {
                            "type": "boolean",
                            "description": "Use models for document processing",
                            "values": ["true", "false"],
                            "value": "true"
                        },
                        "enable_remote_services": {
                            "type": "boolean",
                            "description": "Enable remote services for processing",
                            "values": ["true", "false"],
                            "value": "false"
                        },
                        "do_table_structure": {
                            "type": "boolean",
                            "description": "Enable table structure recognition",
                            "values": ["true", "false"],
                            "value": "true"
                        },
                        "table_structure_mode": {
                            "type": "dropdown",
                            "description": "TableFormer mode for table structure recognition",
                            "values": ["FAST", "ACCURATE"],
                            "value": "ACCURATE"
                        }
                    }
                }
            }
        },
        "Chunker": {
            "selected": chunker,
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
                                ""
                            ],
                            "value": ""
                        }
                    }
                },
                "Sentence": {
                    "name": "Sentence",
                    "type": "Chunker",
                    "description": "Sentence-based chunker",
                    "library": ["nltk"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Sentences": {
                            "type": "number",
                            "description": "Number of sentences per chunk",
                            "values": [],
                            "value": "3"
                        },
                        "Overlap": {
                            "type": "number",
                            "description": "Number of sentences to overlap",
                            "values": [],
                            "value": "1"
                        }
                    }
                }
            }
        },
        "Embedder": {
            "selected": embedder,
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
                                "paraphrase-MiniLM-L6-v2"
                            ],
                            "value": "all-MiniLM-L6-v2"
                        },
                        "vector_dimension": {
                            "type": "number",
                            "value": 384,
                            "description": "Vector dimension for SentenceTransformers",
                            "values": []
                        }
                    },
                    "max_batch_size": 32,
                },
                "Ollama": {
                    "name": "Ollama",
                    "type": "Embedder",
                    "description": "Test embedder using Ollama",
                    "library": ["aiohttp"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Model": {
                            "type": "dropdown",
                            "description": "Select an Ollama model",
                            "values": ["nomic-embed-text", "llama2", "mistral"],
                            "value": "nomic-embed-text"
                        },
                        "Batch Size": {
                            "type": "number",
                            "description": "Number of texts to process in a single batch",
                            "values": [],
                            "value": "32"
                        },
                        "Timeout": {
                            "type": "number",
                            "description": "Timeout in seconds for embedding requests",
                            "values": [],
                            "value": "30"
                        },
                        "Retry Attempts": {
                            "type": "number",
                            "description": "Number of retry attempts for failed requests",
                            "values": [],
                            "value": "3"
                        },
                        "vector_dimension": {
                            "type": "number",
                            "value": 768,
                            "description": "Vector dimension for Ollama",
                            "values": []
                        }
                    },
                    "max_batch_size": 32,
                }
            }
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
        }
    }
    
    return FileConfig(
        filename=str(test_file),
        content=file_content,
        extension=".md",
        file_size=len(file_content),
        labels=["test", "markdown"],
        source="local",
        metadata="Test document for RAG pipeline",
        fileID="test_001",
        isURL=False,
        overwrite=True,
        rag_config=config,
        status="READY",
        status_report={}
    )

async def test_pipeline(reader_name: str, chunker_name: str, embedder_name: str):
    """Test the RAG pipeline with specified reader, chunker, and embedder"""
    print(f"\n=== Testing Pipeline: {reader_name} Reader + {chunker_name} Chunker + {embedder_name} Embedder ===\n")
    
    # Initialize RAG manager
    rag_manager = RAGManager()
    test_file = None
    
    try:
        # Create test file and config
        test_file = create_test_file()
        file_config = create_file_config(test_file, reader_name, chunker_name, embedder_name)
    
        # 1. Connect to Qdrant
        print("1. Connecting to Qdrant...")
        client = await rag_manager.connect("Qdrant", url="localhost", port=6333)
        if not client:
            raise Exception("Failed to connect to Qdrant")
        print("✓ Successfully connected to Qdrant\n")

        # 2. Import document
        print("2. Importing document...")
        collection_name = f"test_collection_{reader_name}_{chunker_name}_{embedder_name}"
        
        # Get vector size from embedder config
        component = file_config.rag_config["Embedder"].components[embedder_name]
        embedder_config = component.config
        vector_dimension = getattr(component, "vector_dimension", None)
        if vector_dimension is not None:
            embedder_config = dict(embedder_config)  # make a copy
            embedder_config["vector_dimension"] = vector_dimension
        vector_size = rag_manager.embedder_manager.embedders[embedder_name].get_vector_size(embedder_config)
        
        # Create collection first
        collection_created = await rag_manager.vector_store_manager.create_collection(
            collection_name,
            vector_size=vector_size
        )
        if not collection_created:
            raise Exception(f"Failed to create collection {collection_name}")
            
        # Import document
        await rag_manager.import_document(
            client,
            file_config,
            collection_name
        )
        print("✓ Document imported successfully\n")

        # 3. Test retrieval
        print("3. Testing retrieval...")
        query = "What are the key components of the RAG pipeline?"
        print(f"Query: {query}")
        
        documents, context = await rag_manager.retrieve_chunks(
            client,
            query,
            file_config.rag_config,
            collection_name,
            labels=file_config.labels
        )
        
        print("\nRetrieved documents:")
        for doc in documents:
            print(f"- {doc['title']}")
        print("\nContext:")
        print(context)

        print("\n✓ Retrieval successful")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\nCleaning up...")
        if client:
            await rag_manager.disconnect()
        if test_file and test_file.exists():
            test_file.unlink()
        print("✓ Cleanup completed")

async def main():
    """Run all test configurations."""
    try:
        # Test all combinations of readers, chunkers, and embedders
        readers = ["Default", "Docling"]
        chunkers = ["Recursive", "Sentence"]
        embedders = ["SentenceTransformers", "Ollama"]
        
        for reader in readers:
            for chunker in chunkers:
                for embedder in embedders:
                    await test_pipeline(reader, chunker, embedder)
        
        print("\n✓ All test configurations completed successfully!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
