from typing import Any, List, Dict, Optional
import asyncio
import weaviate
from weaviate.client import WeaviateAsyncClient
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter, Sort, MetadataQuery
from weaviate.collections.classes.data import DataObject
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.init import AdditionalConfig, Timeout

from rag_backend.components.interfaces import VectorStore
from rag_backend.server.helpers import LoggerManager
from rag_backend.components.interfaces import InputConfig

class WeaviateStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.name = "Weaviate"
        self.description = "Vector store implementation using Weaviate"
        self.type = "VectorStore"
        self.client: Optional[WeaviateAsyncClient] = None
        self.logger = LoggerManager()
        
        # Default configuration
        self.config = {
            "Collection Name": InputConfig(
                type="text",
                value="default_collection",
                description="Name of the collection to store vectors",
                values=[],
            ),
            "Vector Size": InputConfig(
                type="number",
                value=384,  # Default to SentenceTransformers size
                description="Size of the vectors to be stored",
                values=[],
            ),
            "Distance": InputConfig(
                type="dropdown",
                value="Cosine",
                description="Distance metric for vector similarity",
                values=["Cosine", "Euclidean", "Dot"],
            ),
            "Host Config": InputConfig(
                type="dropdown",
                value="docker",
                description="Deployment configuration",
                values=["local", "docker", "cloud"],
            )
        }
    
    async def connect(self, url: str, port: int, config: dict = None):
        """Connect to Weaviate instance."""
        try:
            # Set timeout
            timeout = Timeout(init=60, query=300, insert=300)
            
            # Get host config
            host_config = config.get("Host Config", {}).get("value", "local") if config else "local"
            
            # Initialize client based on host config
            if host_config == "cloud":
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=weaviate.AuthApiKey(api_key=config.get("API Key", {}).get("value")),
                    additional_config=AdditionalConfig(timeout=timeout)
                )
            else:  # local or docker
                self.client = weaviate.connect_to_local(
                    host=url,
                    port=port,
                    additional_config=AdditionalConfig(timeout=timeout)
                )
            
            # Test connection
            if self.client.is_ready():
                await self.logger.send_report(
                    "weaviate_connection",
                    "INFO",
                    f"Successfully connected to Weaviate at {url}:{port}",
                    took=0
                )
                return self.client
            else:
                await self.logger.send_report(
                    "weaviate_connection",
                    "ERROR",
                    "Failed to connect to Weaviate: Client not ready",
                    took=0
                )
                return None
                
        except Exception as e:
            await self.logger.send_report(
                "weaviate_connection",
                "ERROR",
                f"Failed to connect to Weaviate: {str(e)}",
                took=0
            )
            return None

    async def disconnect(self):
        """Disconnect from Weaviate instance."""
        if self.client:
            try:
                self.client.close()
                await self.logger.send_report(
                    "weaviate_connection",
                    "INFO",
                    "Successfully disconnected from Weaviate",
                    took=0
                )
            except Exception as e:
                await self.logger.send_report(
                    "weaviate_connection",
                    "ERROR",
                    f"Error disconnecting from Weaviate: {str(e)}",
                    took=0
                )

    async def create_collection(self, collection_name: str, vector_size: int = 384):
        """Create a new collection in Weaviate."""
        try:
            if not self.client:
                await self.logger.send_report(
                    "weaviate_collection",
                    "ERROR",
                    "No connection to Weaviate",
                    took=0
                )
                return False

            # Create collection configuration
            collection_config = {
                "name": collection_name,
                "vectorizer": "none",
                "vectorIndexConfig": {
                    "distance": "cosine"
                },
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    }
                ]
            }
            
            # Create collection using collections API
            self.client.collections.create(collection_config)
            await self.logger.send_report(
                "weaviate_collection",
                "INFO",
                f"Successfully created collection: {collection_name}",
                took=0
            )
            return True
            
        except Exception as e:
            await self.logger.send_report(
                "weaviate_collection",
                "ERROR",
                f"Failed to create collection {collection_name}: {str(e)}",
                took=0
            )
            return False

    async def delete_collection(self, collection_name: str):
        """Delete a collection from Weaviate."""
        try:
            if not self.client:
                await self.logger.send_report(
                    "weaviate_collection",
                    "ERROR",
                    "No connection to Weaviate",
                    took=0
                )
                return False

            # Delete collection using collections API
            self.client.collections.delete(collection_name)
            await self.logger.send_report(
                "weaviate_collection",
                "INFO",
                f"Successfully deleted collection: {collection_name}",
                took=0
            )
            return True
            
        except Exception as e:
            await self.logger.send_report(
                "weaviate_collection",
                "ERROR",
                f"Failed to delete collection {collection_name}: {str(e)}",
                took=0
            )
            return False

    async def get_collections(self):
        """Get list of all collections."""
        try:
            if not self.client:
                await self.logger.send_report(
                    "weaviate_collection",
                    "ERROR",
                    "No connection to Weaviate",
                    took=0
                )
                return []

            # Get collections using collections API
            collections = self.client.collections.list()
            return [c.name for c in collections]
            
        except Exception as e:
            await self.logger.send_report(
                "weaviate_collection",
                "ERROR",
                f"Failed to get collections: {str(e)}",
                took=0
            )
            return []

    async def insert_vectors(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict], **kwargs) -> List[str]:
        """Insert vectors into Weaviate collection"""
        try:
            if not self.client:
                raise Exception("No active Weaviate client")
            
            if not await self.client.schema.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")
            
            # Prepare objects for batch insertion
            objects = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                objects.append({
                    "class": collection_name,
                    "properties": meta,
                    "vector": vector
                })
            
            # Insert objects
            result = await self.client.batch.add_objects(objects)
            return [str(i) for i in range(len(objects))]
        except Exception as e:
            print(f"Debug: Error inserting vectors: {str(e)}")
            return []
    
    async def search_vectors(self, collection_name: str, query_vector: List[float], limit: int, **kwargs) -> List[Dict]:
        """Search for similar vectors in Weaviate collection"""
        try:
            if not self.client:
                raise Exception("No active Weaviate client")
            
            if not await self.client.schema.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")
            
            # Perform vector search
            result = await self.client.query.get(
                collection_name,
                ["text", "doc_id", "chunk_id", "title", "labels"]
            ).with_near_vector({
                "vector": query_vector
            }).with_limit(limit).do()
            
            # Format results
            return [{
                "id": str(i),
                "score": hit["_additional"]["certainty"],
                "metadata": {
                    "text": hit["text"],
                    "doc_id": hit["doc_id"],
                    "chunk_id": hit["chunk_id"],
                    "title": hit["title"],
                    "labels": hit["labels"]
                }
            } for i, hit in enumerate(result["data"]["Get"][collection_name])]
        except Exception as e:
            print(f"Debug: Error searching vectors: {str(e)}")
            return []
    
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Weaviate collection"""
        try:
            if not self.client:
                raise Exception("No active Weaviate client")
            
            if not await self.client.schema.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")
            
            # Delete objects
            await self.client.batch.delete_objects(
                collection_name,
                vector_ids
            )
            return True
        except Exception as e:
            print(f"Debug: Error deleting vectors: {str(e)}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a Weaviate collection"""
        try:
            if not self.client:
                raise Exception("No active Weaviate client")
            
            if not await self.client.schema.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")
            
            # Get collection info
            schema = await self.client.schema.get()
            collection = next((c for c in schema.classes if c.class_name == collection_name), None)
            
            if not collection:
                return {}
            
            # Get collection statistics
            stats = await self.client.query.aggregate(collection_name).with_meta_count().do()
            
            return {
                "name": collection_name,
                "schema": collection,
                "count": stats["data"]["Aggregate"][collection_name][0]["meta"]["count"]
            }
        except Exception as e:
            print(f"Debug: Error getting collection info: {str(e)}")
            return {} 