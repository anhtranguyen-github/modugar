import weaviate
from weaviate.classes.init import AdditionalConfig, Timeout
import os
from dotenv import load_dotenv
from weaviate import Client

load_dotenv()

def test_connection_methods():
    """Test Weaviate connection and print client version and available methods."""
    print("\n=== Testing Weaviate Connection and API ===")
    
    try:
        # Print Weaviate client version
        print(f"\nWeaviate client version: {weaviate.__version__}")
        
        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            additional_config=AdditionalConfig(timeout=Timeout(init=60, query=300, insert=300))
        )
        
        print("\nClient object type:", type(client))
        print("Available methods:", dir(client))
        
        # Test collections API
        print("\nTesting collections API...")
        print("Collections object type:", type(client.collections))
        print("Collections methods:", dir(client.collections))
        
        # Try to create a test collection using create_from_dict
        try:
            collection_name = "test_collection"
            print(f"\nTrying to create collection: {collection_name}")
            
            # Test creating a collection
            try:
                print("\nTrying to create collection:", collection_name)
                collection_dict = {
                    "name": collection_name,
                    "description": "A test collection for debugging",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"]
                        }
                    ],
                    "vectorizer": "none",
                    "vectorIndexConfig": {
                        "distance": "cosine"
                    }
                }
                print("\nCollection config:", collection_dict)
                client.collections.create_from_dict(collection_dict)
                print("✓ Collection created successfully")
            except Exception as e:
                print(f"Error with collection operations: {str(e)}")
                print(f"Error type: {type(e)}")
                print(f"Error details: {e.__dict__}")
            
            # List collections
            collections = client.collections.list()
            print("\nAvailable collections:", [c.name for c in collections])
            
            # Try to delete the collection
            client.collections.delete(collection_name)
            print("Collection deleted successfully")
        except Exception as e:
            print("Error with collection operations:", str(e))
            print("Error type:", type(e))
            print("Error details:", e.__dict__ if hasattr(e, '__dict__') else 'No details available')
        
        # Close connection
        client.close()
        print("\n✓ Connection test completed")
        
    except Exception as e:
        print(f"\n✗ Connection test failed: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")

if __name__ == "__main__":
    test_connection_methods() 