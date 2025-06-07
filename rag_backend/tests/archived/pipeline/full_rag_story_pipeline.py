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

def create_file_config(story_file, reader="Default", chunker="Recursive", embedder="SentenceTransformers", vector_store="Qdrant"):
    """Create a FileConfig with test configuration."""
    file_content = read_file_as_base64(story_file)
    
    # Set vector dimensions based on embedder
    vector_size = 768 if embedder == "Ollama" else 384  # Ollama uses 768, SentenceTransformers uses 384
    
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
                            "description": "Whether to use Docling models for processing",
                            "values": ["true", "false"],
                            "value": "false"
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
                            "values": ["\n\n", "\n", " ", ".", ",", ""],
                            "value": ""
                        }
                    }
                },
                "Sentence": {
                    "name": "Sentence",
                    "type": "Chunker",
                    "description": "Sentence-based text chunker",
                    "library": ["nltk"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Chunk Size": {
                            "type": "number",
                            "description": "Maximum number of sentences per chunk",
                            "values": [],
                            "value": "3"
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
                            "values": ["all-MiniLM-L6-v2"],
                            "value": "all-MiniLM-L6-v2"
                        }
                    },
                    "max_batch_size": 32
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
                            "values": ["nomic-embed-text"],
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
                        }
                    },
                    "max_batch_size": 32
                }
            }
        },
        "VectorStore": {
            "selected": vector_store,
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
                            "value": "story_collection"
                        },
                        "Vector Size": {
                            "type": "number",
                            "description": "Size of the vectors to be stored",
                            "values": [],
                            "value": str(vector_size)  # Use the correct vector size based on embedder
                        }
                    }
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
        },
        "Generator": {
            "selected": "Ollama",
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
                            "values": ["qwen3:8b"],
                            "value": "qwen3:8b"
                        },
                        "System Message": {
                            "type": "text",
                            "description": "System message for the model",
                            "values": [],
                            "value": "You are a helpful assistant that answers questions about the world of Viremyr and its lore. Use the provided context to give accurate and detailed answers."
                        }
                    }
                }
            }
        }
    }
    
    return FileConfig(
        filename=str(story_file),
        content=file_content,
        extension=".md",
        file_size=len(file_content),
        labels=["story", "lore", "fantasy"],
        source="local",
        metadata="The Lore of Viremyr, the Withered World",
        fileID="story_001",
        isURL=False,
        overwrite=True,
        rag_config=config,
        status="READY",
        status_report={}
    )

async def test_qa(rag_manager, client, rag_config, question, log_func):
    """Test a single question-answer pair"""
    log_func(f"\nQuestion: {question}")
    
    try:
        # Retrieve relevant chunks
        documents, context = await rag_manager.retrieve_chunks(
            client,
            question,
            rag_config,
            "story_collection",
            labels=["story", "lore", "fantasy"]
        )
        
        log_func(f"\nRetrieved context:")
        log_func(context)
        
        # Generate answer
        log_func("\nGenerating answer...")
        async for response in rag_manager.generate_answer(rag_config, question, context, []):
            if response.get("message"):
                log_func(f"Answer: {response['message']}")
            if response.get("finish_reason") == "stop":
                break
                
    except Exception as e:
        log_func(f"Error: {str(e)}")
    
    log_func("-" * 80)

async def test_pipeline(reader_name: str, chunker_name: str, embedder_name: str, vector_store_name: str, log_func):
    """Test a specific pipeline configuration"""
    log_func(f"\n=== Testing Pipeline Configuration ===")
    log_func(f"Reader: {reader_name}")
    log_func(f"Chunker: {chunker_name}")
    log_func(f"Embedder: {embedder_name}")
    log_func(f"Vector Store: {vector_store_name}")
    log_func("=" * 50 + "\n")
    
    rag_manager = RAGManager()
    client = None
    
    try:
        # 1. Connect to vector store
        log_func("1. Connecting to vector store...")
        client = await rag_manager.connect(vector_store_name, url="localhost", port=6333)
        if not client:
            raise Exception(f"Failed to connect to {vector_store_name}")
        log_func(f"✓ Successfully connected to {vector_store_name}\n")

        # 2. Import story document
        log_func("2. Importing story document...")
        story_file = Path("Throne of the Verdant Flame.md")
        file_config = create_file_config(story_file, reader_name, chunker_name, embedder_name, vector_store_name)
        
        # Get vector size from config
        vector_size = int(file_config.rag_config["VectorStore"].components[vector_store_name].config["Vector Size"].value)
        
        # Create collection
        collection_created = await rag_manager.vector_store_manager.create_collection(
            "story_collection",
            vector_size=vector_size
        )
        if not collection_created:
            raise Exception("Failed to create collection")
            
        # Import document
        await rag_manager.import_document(
            client,
            file_config,
            "story_collection"
        )
        log_func("✓ Story imported successfully\n")

        # 3. Test questions
        log_func("3. Testing questions about the story...")
        
        test_questions = [
            "Who is Ael'Thir and what role did they play in the creation of Viremyr?",
            "What is the Verdant Flame and what happened to it?",
            "Who is Mournhal and what did they do?",
            "What are the main factions in the story and what are their goals?",
            "What is the role of the Kindleshade in the story?",
            "What is the Heartroot Pact and what happened to it?",
            "What is the current state of Viremyr and what caused it?",
            "What are the possible outcomes for the Kindleshade's journey?"
        ]

        for i, question in enumerate(test_questions, 1):
            log_func(f"\nTest Case {i}:")
            await test_qa(
                rag_manager,
                client,
                file_config.rag_config,
                question,
                log_func
            )

    except Exception as e:
        log_func(f"\n❌ Error: {str(e)}")
        raise
    finally:
        # Cleanup
        log_func("\nCleaning up...")
        if client:
            await rag_manager.disconnect()  # Remove the client argument
        log_func("✓ Cleanup completed")

async def run_tests():
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"full_rag_pipeline_test_results_{timestamp}.txt"
    error_log_file = f"full_rag_pipeline_errors_{timestamp}.txt"
    
    with open(log_file, "w") as f, open(error_log_file, "w") as error_f:
        def log(message, is_error=False):
            print(message)
            f.write(message + "\n")
            f.flush()
            if is_error:
                error_f.write(message + "\n")
                error_f.flush()

        log(f"\n=== Testing Full RAG Pipeline with All Component Combinations ===\n")
        log(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Define all component combinations to test
        readers = ["Default", "Docling"]
        chunkers = ["Recursive", "Sentence"]
        embedders = ["SentenceTransformers", "Ollama"]
        vector_stores = ["Qdrant", "Weaviate"]  
        
        total_combinations = len(readers) * len(chunkers) * len(embedders) * len(vector_stores)
        current_combination = 0
        failed_combinations = []
        
        for reader in readers:
            for chunker in chunkers:
                for embedder in embedders:
                    for vector_store in vector_stores:
                        current_combination += 1
                        pipeline_name = f"{reader}-{chunker}-{embedder}-{vector_store}"
                        log(f"\nTesting combination {current_combination} of {total_combinations}")
                        log(f"Pipeline: {pipeline_name}")
                        
                        try:
                            await test_pipeline(reader, chunker, embedder, vector_store, log)
                            log(f"✓ Pipeline {pipeline_name} completed successfully")
                        except Exception as e:
                            error_msg = f"❌ Pipeline {pipeline_name} failed: {str(e)}"
                            log(error_msg, is_error=True)
                            failed_combinations.append({
                                "pipeline": pipeline_name,
                                "error": str(e),
                                "components": {
                                    "reader": reader,
                                    "chunker": chunker,
                                    "embedder": embedder,
                                    "vector_store": vector_store
                                }
                            })
                            continue
        
        # Summary report
        log("\n=== Test Summary ===")
        log(f"Total combinations tested: {total_combinations}")
        log(f"Successful combinations: {total_combinations - len(failed_combinations)}")
        log(f"Failed combinations: {len(failed_combinations)}")
        
        if failed_combinations:
            log("\n=== Failed Pipelines ===")
            for failure in failed_combinations:
                log(f"\nPipeline: {failure['pipeline']}")
                log(f"Components:")
                log(f"  - Reader: {failure['components']['reader']}")
                log(f"  - Chunker: {failure['components']['chunker']}")
                log(f"  - Embedder: {failure['components']['embedder']}")
                log(f"  - Vector Store: {failure['components']['vector_store']}")
                log(f"Error: {failure['error']}")
                log("-" * 50)
        
        log(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Full results saved to: {log_file}")
        log(f"Error details saved to: {error_log_file}")

if __name__ == "__main__":
    asyncio.run(run_tests()) 