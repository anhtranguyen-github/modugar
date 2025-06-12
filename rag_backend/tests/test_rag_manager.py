import asyncio
import os
from pathlib import Path
import base64
from datetime import datetime
from rag_backend.rag_manager import RAGManager
from rag_backend.server.types import FileConfig
from rag_backend.server.helpers import LoggerManager
from rag_backend.components.embedding.OpenAIEmbedder import OpenAIEmbedder

def read_file_as_base64(path):
    """Read a file as bytes and return a base64-encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def create_test_file():
    """Use Throne of the Verdant Flame.md as the test file."""
    test_file = Path("data/Throne of the Verdant Flame.md")
    if not test_file.exists():
        raise Exception(f"Test file not found: {test_file}")
    return test_file

def create_file_config(
    test_file,
    reader_name: str,
    chunker_name: str,
    embedder_name: str,
    vector_store_name: str,
    retriever_name: str,
    generator_name: str,
) -> FileConfig:
    """Create a FileConfig with test configuration."""
    file_content = read_file_as_base64(test_file)
    
    # Set vector dimensions based on embedder
    vector_size = 768 if embedder_name == "Ollama" else (1536 if embedder_name == "OpenAI" else 384)  # Ollama uses 768, OpenAI uses 1536, SentenceTransformers uses 384
    
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
                },
                "Sentence": {
                    "name": "Sentence",
                    "type": "Chunker",
                    "description": "Sentence-based text chunker",
                    "library": ["nltk"],
                    "available": True,
                    "variables": [],
                    "config": {
                        "Sentences": {
                            "type": "number",
                            "description": "Maximum number of sentences per chunk",
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
            "selected": embedder_name,
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
                        },
                        "vector_dimension": {
                            "type": "number",
                            "value": 384,
                            "description": "Vector dimension for SentenceTransformers",
                            "values": []
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
                        },
                        "vector_dimension": {
                            "type": "number",
                            "value": 768,
                            "description": "Vector dimension for Ollama",
                            "values": []
                        }
                    },
                    "max_batch_size": 32
                },
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
                "Weaviate": {
                    "name": "Weaviate",
                    "type": "VectorStore",
                    "description": "Weaviate vector store for storing and retrieving embeddings",
                    "library": ["weaviate-client"],
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
                },
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
                },
                "OpenAI": {
                    "name": "OpenAI",
                    "type": "Generator",
                    "description": "Generate answers using OpenAI",
                    "library": ["httpx"],
                    "available": True,
                    "variables": ["OPENAI_API_KEY"],
                    "config": {
                        "Model": {
                            "type": "dropdown",
                            "description": "Select an OpenAI model",
                            "values": ["gpt-3.5-turbo", "gpt-4"],
                            "value": "gpt-3.5-turbo"
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
        filename=str(test_file),
        content=file_content,
        extension=".md",
        file_size=test_file.stat().st_size,
        labels=["test", "markdown"],
        source="local",
        metadata="Test document for RAG pipeline",
        fileID=f"test_{reader_name}_{chunker_name}_{embedder_name}",
        isURL=False,
        overwrite=True,
        rag_config=config,
        status="READY",
        status_report={}
    )

async def check_ollama_connection():
    """Check if Ollama is running and accessible."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                return response.status == 200
    except:
        return False

async def test_qa(rag_manager, client, file_config, question, log_func):
    """Test a single question-answer pair"""
    log_func(f"\nQuestion: {question}")
    
    try:
        # Get collection name from config
        collection_name = file_config.rag_config["VectorStore"].components["Weaviate"].config["Collection Name"].value
        
        # Retrieve relevant chunks
        documents, context = await rag_manager.retrieve_chunks(
            client,
            question,
            file_config.rag_config,
            collection_name,
            labels=["test", "markdown"]
        )
        
        log_func(f"\nRetrieved context:")
        log_func(context)
        
        # Generate answer
        log_func("\nGenerating answer...")
        answer = ""
        try:
            # Create a list to store all responses
            responses = []
            async for response in rag_manager.generate_answer(question, file_config, context):
                responses.append(response)
                if isinstance(response, dict):
                    if response.get("message"):
                        answer += response["message"]
                        log_func(f"Answer: {response['message']}", is_answer=True)
                    if response.get("finish_reason") == "stop":
                        break
                elif isinstance(response, str):
                    answer += response
                    log_func(f"Answer: {response}", is_answer=True)
        except Exception as e:
            log_func(f"Error during generation: {str(e)}", is_error=True)
            raise
        finally:
            # Ensure we've processed all responses
            for response in responses:
                if isinstance(response, dict) and response.get("finish_reason") == "stop":
                    break
                
    except Exception as e:
        log_func(f"Error: {str(e)}", is_error=True)
    
    log_func("-" * 80)

async def test_pipeline(reader_name: str, chunker_name: str, embedder_name: str, generator_name: str, log_func):
    """Test a specific pipeline configuration"""
    log_func(f"\n=== Testing Pipeline Configuration ===")
    log_func(f"Reader: {reader_name}")
    log_func(f"Chunker: {chunker_name}")
    log_func(f"Embedder: {embedder_name}")
    log_func(f"Generator: {generator_name}")
    log_func("=" * 50 + "\n")
    
    # Check Ollama connection if using Ollama generator
    if generator_name == "Ollama":
        log_func("Checking Ollama connection...")
        if not await check_ollama_connection():
            raise Exception("Ollama is not running or not accessible at localhost:11434. Please start Ollama and try again.")
        log_func("✓ Ollama connection successful")
    
    rag_manager = RAGManager()
    test_file = None
    client = None
    
    try:
        # Create test file and config
        test_file = create_test_file()
        file_config = create_file_config(
            test_file,
            reader_name,
            chunker_name,
            embedder_name,
            "Qdrant",
            "Advanced",
            generator_name
        )
        
        # Get vector size from embedder config
        vector_dimension = file_config.rag_config["Embedder"].components[embedder_name].config["vector_dimension"].value
        log_func(f"Using vector dimension: {vector_dimension}")
        
        # 1. Connect to Qdrant
        log_func("1. Connecting to Qdrant...")
        client = await rag_manager.connect(
            "Qdrant",
            url="localhost",
            port=6333,
            key="",  # Empty key for local connection
            config={
                "Host Config": {"value": "local"},
                "Collection Name": {"value": "test_collection"},
                "Vector Size": {"value": str(vector_dimension)},  # Use the actual vector dimension from embedder
                "Distance": {"value": "Cosine"}
            }
        )
        if not client:
            raise Exception("Failed to connect to Qdrant")
        log_func("✓ Successfully connected to Qdrant\n")

        # 2. Import document
        log_func("2. Importing document...")
        collection_name = "test_collection"  # Use consistent collection name
        
        # Create collection with the correct vector size
        collection_created = await rag_manager.vector_store_manager.create_collection(
            "Qdrant",  # store_name
            client,    # client
            collection_name,
            vector_size=int(vector_dimension)  # Use the actual vector dimension
        )
        if not collection_created:
            raise Exception(f"Failed to create collection {collection_name}")
            
        # Import document
        await rag_manager.import_document(
            client,
            file_config,
            collection_name
        )
        log_func("✓ Document imported successfully\n")

        # 3. Test questions
        log_func("3. Testing questions about the Throne of the Verdant Flame...")
        
        test_questions = [
            "What is the Throne of the Verdant Flame?",
            "What are the key features or characteristics of the throne?",
            "What is the significance or importance of the throne?",
            "What materials or elements is the throne made of?",
            "What is the history or origin of the throne?",
            "What powers or abilities are associated with the throne?",
            "Who can use or access the throne?",
            "What are the main themes or symbolism in the throne's description?"
        ]

        for i, question in enumerate(test_questions, 1):
            log_func(f"\nTest Case {i}:")
            await test_qa(
                rag_manager,
                client,
                file_config,
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
            try:
                await rag_manager.disconnect()
            except Exception as e:
                log_func(f"Warning: Error during disconnect: {str(e)}")
        # Don't delete the test file since we want to keep it
        # if test_file and test_file.exists():
        #     test_file.unlink()
        log_func("✓ Cleanup completed")

async def main():
    """Run all tests"""
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create log directories
    log_dirs = {
        "all": Path("logs/all"),
        "errors": Path("logs/errors"),
        "answers": Path("logs/generated_answers")
    }
    
    for dir_path in log_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create log files
    log_files = {
        "all": log_dirs["all"] / f"full_rag_pipeline_test_results_{timestamp}.txt",
        "errors": log_dirs["errors"] / f"full_rag_pipeline_errors_{timestamp}.txt",
        "answers": log_dirs["answers"] / f"generated_answers_{timestamp}.txt"
    }

    with open(log_files["all"], "w") as f, \
         open(log_files["errors"], "w") as error_f, \
         open(log_files["answers"], "w") as answers_f:
        
        def log(message, is_error=False, is_answer=False):
            print(message)
            f.write(message + "\n")
            f.flush()
            if is_error:
                error_f.write(message + "\n")
                error_f.flush()
            if is_answer:
                answers_f.write(message + "\n")
                answers_f.flush()

        log(f"\n=== Testing Full RAG Pipeline with All Component Combinations ===\n")
        log(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Component options
        # readers = ["Default", "Docling"]
        # chunkers = ["Recursive", "Sentence"]
        # embedders = ["SentenceTransformers", "Ollama", "OpenAI"]
        
        readers = ["Default"]
        chunkers = ["Recursive"]
        embedders = ["Ollama"]  # Using OpenAI embedder
        generators = ["Ollama"]  # Test both generators

        # Test all combinations
        total_combinations = len(readers) * len(chunkers) * len(embedders) * len(generators)
        current_combination = 0
        successful_combinations = []
        failed_combinations = []

        log(f"\nTesting {total_combinations} combinations...")

        for reader in readers:
            for chunker in chunkers:
                for embedder in embedders:
                    for generator in generators:
                        current_combination += 1
                        pipeline_name = f"{reader}-{chunker}-{embedder}-{generator}"
                        log(f"\nTesting combination {current_combination}/{total_combinations}")
                        log(f"Pipeline: {pipeline_name}")

                        try:
                            await test_pipeline(reader, chunker, embedder, generator, log)
                            successful_combinations.append(pipeline_name)
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
                                    "generator": generator
                                }
                            })

        # Print summary
        log("\n=== Test Summary ===")
        log(f"Total combinations tested: {total_combinations}")
        log(f"Successful combinations: {len(successful_combinations)}")
        log(f"Failed combinations: {len(failed_combinations)}")

        if failed_combinations:
            log("\n=== Failed Pipelines ===")
            for failure in failed_combinations:
                log(f"\nPipeline: {failure['pipeline']}")
                log(f"Components:")
                log(f"  - Reader: {failure['components']['reader']}")
                log(f"  - Chunker: {failure['components']['chunker']}")
                log(f"  - Embedder: {failure['components']['embedder']}")
                log(f"  - Generator: {failure['components']['generator']}")
                log(f"Error: {failure['error']}")
                log("-" * 50)

        log(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Full results saved to: {log_files['all']}")
        log(f"Error details saved to: {log_files['errors']}")
        log(f"Generated answers saved to: {log_files['answers']}")

if __name__ == "__main__":
    asyncio.run(main()) 