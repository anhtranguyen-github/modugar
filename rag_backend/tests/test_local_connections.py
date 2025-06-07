import asyncio
import os
import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
from dotenv import load_dotenv
from rag_backend.components.vector_store.QdrantStore import QdrantStore
from rag_backend.components.vector_store.WeaviateStore import WeaviateStore
from rag_backend.server.helpers import LoggerManager

async def test_qdrant_local():
    """Test local connection to Qdrant."""
    print("\n=== Testing Qdrant Local Connection ===")
    qdrant = QdrantStore()
    
    try:
        # Test local connection
        print("\nTesting local connection...")
        client = await qdrant.connect(
            url="localhost",
            port=6333,
            config={"Host Config": {"value": "local"}}
        )
        
        if client:
            print("✓ Local connection successful")
            
            # Test collection operations
            collection_name = "test_collection"
            print(f"\nTesting collection operations for {collection_name}...")
            
            # Create collection
            created = await qdrant.create_collection(
                collection_name,
                vector_size=384
            )
            print(f"Collection creation: {'✓' if created else '✗'}")
            
            # Get collections
            collections = await client.get_collections()
            print(f"Available collections: {[c.name for c in collections.collections]}")
            
            # Delete collection
            deleted = await qdrant.delete_collection(collection_name)
            print(f"Collection deletion: {'✓' if deleted else '✗'}")
            
            # Disconnect
            await qdrant.disconnect()
            print("\n✓ All tests completed successfully")
        else:
            print("✗ Local connection failed")
            
    except Exception as e:
        print(f"\n✗ Local connection test failed: {str(e)}")
        raise

async def test_weaviate_local():
    """Test local connection to Weaviate."""
    print("\n=== Testing Weaviate Local Connection ===")
    weaviate_store = WeaviateStore()
    
    try:
        # Test local connection
        print("\nTesting local connection...")
        client = await weaviate_store.connect(
            url="localhost",
            port=8080,
            config={"Host Config": {"value": "local"}}
        )
        
        if client:
            print("✓ Local connection successful")
            
            # Test collection operations
            collection_name = "test_collection"
            print(f"\nTesting collection operations for {collection_name}...")
            
            # Create collection
            created = await weaviate_store.create_collection(
                collection_name,
                vector_size=384
            )
            print(f"Collection creation: {'✓' if created else '✗'}")
            
            # Get collections
            collections = await weaviate_store.get_collections()
            print(f"Available collections: {collections}")
            
            # Delete collection
            deleted = await weaviate_store.delete_collection(collection_name)
            print(f"Collection deletion: {'✓' if deleted else '✗'}")
            
            # Disconnect
            await weaviate_store.disconnect()
            print("\n✓ All tests completed successfully")
        else:
            print("✗ Local connection failed")
            
    except Exception as e:
        print(f"\n✗ Local connection test failed: {str(e)}")
        raise

async def main():
    """Run all local connection tests."""
    try:
        # Test Qdrant local connection
        await test_qdrant_local()
        
        # Test Weaviate local connection
        await test_weaviate_local()
        
        print("\n✓ All local connection tests completed successfully!")
    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 