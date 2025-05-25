from typing import Any, List, Dict
import weaviate
from weaviate.client import WeaviateAsyncClient
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import Filter

from rag_backend.components.interfaces import VectorStore
from rag_backend.server.helpers import LoggerManager

class WeaviateStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.name = "Weaviate"
        self.description = "Vector store implementation using Weaviate"
        self.type = "VectorStore"
        self.client: WeaviateAsyncClient = None
        self.logger = LoggerManager()
    
    async def connect(self, **kwargs) -> Any:
        """Connect to Weaviate instance"""
        try:
            deployment = kwargs.get('deployment', 'Weaviate')
            weaviate_url = kwargs.get('weaviate_url', '')
            weaviate_key = kwargs.get('weaviate_key', '')
            port = kwargs.get('port', '8080')
            
            if deployment == "Weaviate":
                if not weaviate_url or not weaviate_key:
                    raise Exception("Weaviate URL and API key are required for Weaviate deployment")
                self.client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=weaviate_url,
                    auth_credentials=AuthApiKey(weaviate_key),
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=60, query=300, insert=300)
                    ),
                )
            elif deployment == "Docker":
                self.client = weaviate.use_async_with_local(
                    host="weaviate",
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=60, query=300, insert=300)
                    ),
                )
            elif deployment == "Custom":
                if not weaviate_url:
                    raise Exception("Host URL is required for Custom deployment")
                if weaviate_key:
                    self.client = weaviate.use_async_with_local(
                        host=weaviate_url,
                        port=int(port),
                        skip_init_checks=True,
                        auth_credentials=AuthApiKey(weaviate_key),
                        additional_config=AdditionalConfig(
                            timeout=Timeout(init=60, query=300, insert=300)
                        ),
                    )
                else:
                    self.client = weaviate.use_async_with_local(
                        host=weaviate_url,
                        port=int(port),
                        skip_init_checks=True,
                        additional_config=AdditionalConfig(
                            timeout=Timeout(init=60, query=300, insert=300)
                        ),
                    )
            
            await self.client.connect()
            if await self.client.is_ready():
                return self.client
            return None
            
        except Exception as e:
            raise Exception(f"Failed to connect to Weaviate: {str(e)}")
    
    async def disconnect(self) -> bool:
        """Disconnect from Weaviate"""
        try:
            if self.client:
                await self.client.close()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to disconnect from Weaviate: {str(e)}")
    
    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new collection in Weaviate"""
        try:
            if not await self.client.collections.exists(collection_name):
                await self.client.collections.create(name=collection_name)
                return True
            return True
        except Exception as e:
            raise Exception(f"Failed to create collection {collection_name}: {str(e)}")
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Weaviate"""
        try:
            if await self.client.collections.exists(collection_name):
                await self.client.collections.delete(collection_name)
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to delete collection {collection_name}: {str(e)}")
    
    async def insert_vectors(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict], **kwargs) -> List[str]:
        """Insert vectors into Weaviate collection"""
        try:
            collection = self.client.collections.get(collection_name)
            response = await collection.data.insert_many(
                [{"vector": vector, **meta} for vector, meta in zip(vectors, metadata)]
            )
            return list(response.uuids.values())
        except Exception as e:
            raise Exception(f"Failed to insert vectors into {collection_name}: {str(e)}")
    
    async def search_vectors(self, collection_name: str, query_vector: List[float], limit: int, **kwargs) -> List[Dict]:
        """Search for similar vectors in Weaviate collection"""
        try:
            collection = self.client.collections.get(collection_name)
            filters = kwargs.get('filters', None)
            
            if filters:
                weaviate_filters = Filter.by_property(filters['property']).contains_all(filters['values'])
            else:
                weaviate_filters = None
                
            results = await collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=weaviate_filters
            )
            
            return [
                {
                    'id': obj.uuid,
                    'score': obj.metadata.score,
                    'metadata': obj.properties
                }
                for obj in results.objects
            ]
        except Exception as e:
            raise Exception(f"Failed to search vectors in {collection_name}: {str(e)}")
    
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Weaviate collection"""
        try:
            collection = self.client.collections.get(collection_name)
            await collection.data.delete_many(
                where=Filter.by_property("id").contains_any(vector_ids)
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to delete vectors from {collection_name}: {str(e)}")
    
    async def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a Weaviate collection"""
        try:
            collection = self.client.collections.get(collection_name)
            info = await collection.aggregate.over_all(total_count=True)
            return {
                'name': collection_name,
                'total_objects': info.total_count,
                'schema': await collection.config.get()
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info for {collection_name}: {str(e)}") 