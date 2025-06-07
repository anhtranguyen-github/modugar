import asyncio
import os
from dotenv import load_dotenv
from rag_backend.components.vector_store.QdrantStore import QdrantStore
from rag_backend.components.vector_store.WeaviateStore import WeaviateStore
from rag_backend.server.helpers import LoggerManager

# Load environment variables
load_dotenv()

async def test_qdrant_connection():
    """Test Qdrant connection with different configurations."""
    print("\n=== Testing Qdrant Connection ===")
    
    # Test local connection
    print("\nTesting local connection...")
    try:
        qdrant = QdrantStore()
        client = await qdrant.connect(
            url="localhost",
            port=6333,
            config={"Host Config": {"value": "local"}}
        )
        print("Local connection successful!")
        
        # Test collection operations
        collection_name = "test_collection"
        await qdrant.create_collection(collection_name, vector_size=384)
        print(f"Created collection: {collection_name}")
        
        # Get collections
        collections = await client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")
        
        # Cleanup
        await qdrant.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
        
        await qdrant.disconnect()
        print("Disconnected successfully")
    except Exception as e:
        print(f"Local connection test failed: {str(e)}")
    
    # Test Docker connection
    print("\nTesting Docker connection...")
    try:
        qdrant = QdrantStore()
        client = await qdrant.connect(
            url="qdrant",
            port=6334,
            config={"Host Config": {"value": "docker"}}
        )
        print("Docker connection successful!")
        await qdrant.disconnect()
        print("Disconnected successfully")
    except Exception as e:
        print(f"Docker connection test failed: {str(e)}")

async def test_weaviate_connection():
    """Test Weaviate connection with different configurations."""
    print("\n=== Testing Weaviate Connection ===")
    
    # Test local connection
    print("\nTesting local connection...")
    try:
        weaviate = WeaviateStore()
        client = await weaviate.connect(
            url="localhost",
            port=8080,
            config={"Host Config": {"value": "local"}}
        )
        print("Local connection successful!")
        
        # Test collection operations
        collection_name = "TestCollection"
        await weaviate.create_collection(collection_name, vector_size=384)
        print(f"Created collection: {collection_name}")
        
        # Get schema
        schema = client.schema.get()
        print(f"Available classes: {[c['class'] for c in schema['classes']]}")
        
        # Cleanup
        await weaviate.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
        
        await weaviate.disconnect()
        print("Disconnected successfully")
    except Exception as e:
        print(f"Local connection test failed: {str(e)}")
    
    # Test Docker connection
    print("\nTesting Docker connection...")
    try:
        weaviate = WeaviateStore()
        client = await weaviate.connect(
            url="weaviate",
            port=8080,
            config={"Host Config": {"value": "docker"}}
        )
        print("Docker connection successful!")
        await weaviate.disconnect()
        print("Disconnected successfully")
    except Exception as e:
        print(f"Docker connection test failed: {str(e)}")

async def test_docker_containers():
    """Check Docker container status and configuration."""
    print("\n=== Checking Docker Containers ===")
    
    # Check if containers are running
    import subprocess
    
    print("\nChecking container status...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant,weaviate"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error checking containers: {str(e)}")
    
    # Check container networks
    print("\nChecking container networks...")
    try:
        result = subprocess.run(
            ["docker", "network", "ls"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Error checking networks: {str(e)}")
    
    # Check container logs
    print("\nChecking container logs...")
    for container in ["qdrant", "weaviate"]:
        try:
            result = subprocess.run(
                ["docker", "logs", container, "--tail", "10"],
                capture_output=True,
                text=True
            )
            print(f"\n{container.upper()} logs:")
            print(result.stdout)
        except Exception as e:
            print(f"Error checking {container} logs: {str(e)}")

async def main():
    """Run all tests."""
    # Check Docker containers first
    await test_docker_containers()
    
    # Test vector store connections
    await test_qdrant_connection()
    await test_weaviate_connection()

if __name__ == "__main__":
    asyncio.run(main()) 