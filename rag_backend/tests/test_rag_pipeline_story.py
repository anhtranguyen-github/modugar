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

def create_file_config(story_file):
    """Create a FileConfig with test configuration."""
    file_content = read_file_as_base64(story_file)
    
    config = {
        "Reader": {
            "selected": "Default",
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
            "selected": "Recursive",
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
            "selected": "Ollama",
            "components": {
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

async def run_tests():
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"story_qa_test_results_{timestamp}.txt"
    
    with open(log_file, "w") as f:
        def log(message):
            print(message)
            f.write(message + "\n")
            f.flush()

        log(f"\n=== Testing RAG Pipeline with Throne of the Verdant Flame ===\n")
        log(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        rag_manager = RAGManager()
        client = None
        
        try:
            # 1. Connect to Qdrant
            log("1. Connecting to Qdrant...")
            client = await rag_manager.connect("Qdrant", url="localhost", port=6333)
            if not client:
                raise Exception("Failed to connect to Qdrant")
            log("✓ Successfully connected to Qdrant\n")

            # 2. Import story document
            log("2. Importing story document...")
            story_file = Path("Throne of the Verdant Flame.md")
            file_config = create_file_config(story_file)
            
            # Create collection
            collection_created = await rag_manager.vector_store_manager.create_collection(
                "story_collection",
                vector_size=768  # Using Ollama's vector size
            )
            if not collection_created:
                raise Exception("Failed to create collection")
                
            # Import document
            await rag_manager.import_document(
                client,
                file_config,
                "story_collection"
            )
            log("✓ Story imported successfully\n")

            # 3. Test questions
            log("3. Testing questions about the story...")
            
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
                log(f"\nTest Case {i}:")
                await test_qa(
                    rag_manager,
                    client,
                    file_config.rag_config,
                    question,
                    log
                )

        except Exception as e:
            log(f"\n❌ Error: {str(e)}")
            raise
        finally:
            # Cleanup
            log("\nCleaning up...")
            if client:
                await rag_manager.disconnect()
            log("✓ Cleanup completed")
            
        log(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Results saved to: {log_file}")

if __name__ == "__main__":
    asyncio.run(run_tests()) 