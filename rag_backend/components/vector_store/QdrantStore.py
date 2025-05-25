from typing import Any, List, Dict
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

from rag_backend.components.interfaces import VectorStore
from rag_backend.server.helpers import LoggerManager

class QdrantStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.name = "Qdrant"
        self.description = "Vector store implementation using Qdrant"
        self.type = "VectorStore"
        self.client: AsyncQdrantClient = None
        self.logger = LoggerManager()
    
    async def connect(self, **kwargs) -> Any:
        """Connect to Qdrant instance"""
        try:
            url = kwargs.get('url', 'localhost')
            port = kwargs.get('port', 6333)
            api_key = kwargs.get('api_key', None)
            prefer_grpc = kwargs.get('prefer_grpc', True)
            
            self.client = AsyncQdrantClient(
                url=url,
                port=port,
                api_key=api_key,
                prefer_grpc=prefer_grpc
            )
            
            # Test connection
            await self.client.get_collections()
            return self.client
            
        except Exception as e:
            raise Exception(f"Failed to connect to Qdrant: {str(e)}")
    
    async def disconnect(self) -> bool:
        """Disconnect from Qdrant"""
        try:
            if self.client:
                await self.client.close()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to disconnect from Qdrant: {str(e)}")
    
    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new collection in Qdrant"""
        try:
            vector_size = kwargs.get('vector_size', 1536)  # Default to OpenAI embedding size
            distance = kwargs.get('distance', Distance.COSINE)
            
            # Check if collection exists
            collections = await self.client.get_collections()
            if collection_name in [col.name for col in collections.collections]:
                return True
            
            # Create collection
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            return True
            
        except Exception as e:
            raise Exception(f"Failed to create collection {collection_name}: {str(e)}")
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant"""
        try:
            collections = await self.client.get_collections()
            if collection_name in [col.name for col in collections.collections]:
                await self.client.delete_collection(collection_name)
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to delete collection {collection_name}: {str(e)}")
    
    async def insert_vectors(self, collection_name: str, vectors: List[List[float]], metadata: List[Dict], **kwargs) -> List[str]:
        """Insert vectors into Qdrant collection"""
        try:
            # Generate proper UUIDs if not provided
            ids = kwargs.get('ids', [str(uuid.uuid4()) for _ in range(len(vectors))])
            
            # Create points
            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=meta
                )
                for id_, vector, meta in zip(ids, vectors, metadata)
            ]
            
            # Insert points
            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            return ids
            
        except Exception as e:
            raise Exception(f"Failed to insert vectors into {collection_name}: {str(e)}")
    
    async def search_vectors(self, collection_name: str, query_vector: List[float], limit: int, **kwargs) -> List[Dict]:
        """Search for similar vectors in Qdrant collection"""
        try:
            filters = kwargs.get('filters', None)
            score_threshold = kwargs.get('score_threshold', None)
            
            # Convert filters to Qdrant format
            qdrant_filter = None
            if filters:
                from qdrant_client.http import models as rest
                conditions = []
                
                if "must" in filters:
                    for condition in filters["must"]:
                        if "key" in condition and "match" in condition:
                            field_name = condition["key"]
                            match_value = condition["match"]
                            if "any" in match_value:
                                conditions.append(
                                    rest.FieldCondition(
                                        key=field_name,
                                        match=rest.MatchAny(any=match_value["any"])
                                    )
                                )
                
                if conditions:
                    qdrant_filter = rest.Filter(
                        must=conditions
                    )
            
            # Perform search
            search_result = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold
            )
            
            # Format results
            return [
                {
                    'id': str(point.id),
                    'score': point.score,
                    'metadata': point.payload
                }
                for point in search_result
            ]
            
        except Exception as e:
            raise Exception(f"Failed to search vectors in {collection_name}: {str(e)}")
    
    async def delete_vectors(self, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Qdrant collection"""
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids
                )
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to delete vectors from {collection_name}: {str(e)}")
    
    async def get_collection_info(self, collection_name: str) -> Dict:
        """Get information about a Qdrant collection"""
        try:
            collection_info = await self.client.get_collection(collection_name)
            collection_stats = await self.client.get_collection(collection_name)
            
            return {
                'name': collection_name,
                'total_objects': collection_stats.points_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance,
                'status': collection_info.status
            }
        except Exception as e:
            raise Exception(f"Failed to get collection info for {collection_name}: {str(e)}") 