import asyncio
import os
from datetime import datetime, timedelta
from rag_backend.rag_manager import ClientManager
from rag_backend.server.types import Credentials

def log(message):
    """Simple logging function that prints with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

async def test_client_connection():
    """Test basic client connection and caching"""
    log("Starting client connection test...")
    
    # Create test credentials
    credentials1 = Credentials(
        deployment="test1",
        url="http://localhost:6333",
        key="test_key_1"
    )
    
    credentials2 = Credentials(
        deployment="test2",
        url="http://localhost:6334",
        key="test_key_2"
    )
    
    client_manager = ClientManager()
    
    try:
        # Test first connection
        log("Testing first connection...")
        client1 = await client_manager.connect(credentials1)
        log(f"First client connected: {client1 is not None}")
        
        # Test connection caching
        log("\nTesting connection caching...")
        client1_cached = await client_manager.connect(credentials1)
        log(f"Retrieved cached client: {client1_cached is not None}")
        log(f"Same client instance: {client1 is client1_cached}")
        
        # Test second connection
        log("\nTesting second connection...")
        client2 = await client_manager.connect(credentials2)
        log(f"Second client connected: {client2 is not None}")
        log(f"Different client instances: {client1 is not client2}")
        
        # Test client count
        log("\nTesting client count...")
        log(f"Number of connected clients: {len(client_manager.clients)}")
        
        # Test client cleanup
        log("\nTesting client cleanup...")
        # Modify timestamp to simulate old connection
        for cred_hash in client_manager.clients:
            client_manager.clients[cred_hash]["timestamp"] = datetime.now() - timedelta(minutes=11)
        
        await client_manager.clean_up()
        log(f"Number of clients after cleanup: {len(client_manager.clients)}")
        
        # Test disconnection
        log("\nTesting client disconnection...")
        await client_manager.disconnect()
        log(f"Number of clients after disconnection: {len(client_manager.clients)}")
        
    except Exception as e:
        log(f"Error during test: {str(e)}")
        raise
    finally:
        # Ensure cleanup
        await client_manager.disconnect()

async def test_concurrent_connections():
    """Test concurrent connection requests"""
    log("\nStarting concurrent connection test...")
    
    credentials = Credentials(
        deployment="test_concurrent",
        url="http://localhost:6333",
        key="test_key"
    )
    
    client_manager = ClientManager()
    
    try:
        # Create multiple concurrent connection requests
        log("Creating concurrent connection requests...")
        tasks = [
            client_manager.connect(credentials)
            for _ in range(5)
        ]
        
        # Wait for all connections
        clients = await asyncio.gather(*tasks)
        
        # Verify results
        log(f"Number of successful connections: {len(clients)}")
        log(f"Number of unique client instances: {len(set(clients))}")
        log(f"Number of cached clients: {len(client_manager.clients)}")
        
    except Exception as e:
        log(f"Error during concurrent test: {str(e)}")
        raise
    finally:
        await client_manager.disconnect()

async def test_connection_timeout():
    """Test connection timeout handling"""
    log("\nStarting connection timeout test...")
    
    credentials = Credentials(
        deployment="test_timeout",
        url="http://invalid:6333",  # Invalid URL to force timeout
        key="test_key"
    )
    
    client_manager = ClientManager()
    
    try:
        log("Attempting connection with invalid URL...")
        await client_manager.connect(credentials)
    except Exception as e:
        log(f"Expected error occurred: {str(e)}")
    finally:
        await client_manager.disconnect()

async def run_tests():
    """Run all client manager tests"""
    log("=== Starting ClientManager Tests ===\n")
    
    try:
        await test_client_connection()
        await test_concurrent_connections()
        await test_connection_timeout()
        
        log("\n=== All tests completed successfully ===")
    except Exception as e:
        log(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_tests()) 