from typing import Any, List, Dict, Optional
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.grpc import Filter, FieldCondition, Match, Range, Condition
import uuid
import asyncio

from rag_backend.components.interfaces import VectorStore
from rag_backend.server.helpers import LoggerManager
from rag_backend.components.interfaces import InputConfig

class QdrantStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.name = "Qdrant"
        self.description = "Vector store implementation using Qdrant"
        self.type = "VectorStore"
        self.client: Optional[AsyncQdrantClient] = None
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
                value="local",
                description="Deployment configuration",
                values=["local", "docker", "cloud"],
            )
        }
    
    async def connect(self, **kwargs) -> Any:
        """Connect to Qdrant."""
        try:
            print(f"Debug: QdrantStore.connect called with kwargs: {kwargs}")
            
            # Get configuration
            url = kwargs.get("url", "localhost")
            port = kwargs.get("port", "6333")
            api_key = kwargs.get("api_key", "")
            config = kwargs.get("config", {})
            
            # Get host configuration
            host_config = config.get("Host Config", {}).get("value", "local")  # Default to local
            print(f"Debug: Host config: {host_config}")
            
            # Set timeout
            timeout = 60  # 60 seconds timeout
            
            # Log connection attempt
            await self.logger.send_report(
                "qdrant_connection",
                "INFO",
                f"Attempting to connect to Qdrant with configuration: - Host Config: {host_config} - URL: {url} - Port: {port} - API Key provided: {bool(api_key)}",
                took=0
            )
            
            # Initialize client based on host configuration
            if host_config == "cloud":
                if not url or not api_key:
                    raise ValueError("URL and API key are required for cloud deployment")
                print(f"Debug: Initializing cloud connection to {url}")
                await self.logger.send_report(
                    "qdrant_connection",
                    "INFO",
                    "Initializing cloud connection...",
                    took=0
                )
                self.client = AsyncQdrantClient(
                    url=url,
                    api_key=api_key,
                    timeout=timeout
                )
            elif host_config == "docker":
                print(f"Debug: Initializing Docker connection to {url}:{port}")
                await self.logger.send_report(
                    "qdrant_connection",
                    "INFO",
                    "Initializing Docker connection...",
                    took=0
                )
                # For Docker, use the service name as host and default port
                self.client = AsyncQdrantClient(
                    host="qdrant",  # Use the service name from docker-compose
                    port=6333,  # Use the default Qdrant port
                    timeout=timeout
                )
            else:  # local
                print(f"Debug: Initializing local connection to {url}:{port}")
                await self.logger.send_report(
                    "qdrant_connection",
                    "INFO",
                    "Initializing local connection...",
                    took=0
                )
                self.client = AsyncQdrantClient(
                    host=url,
                    port=int(port),
                    timeout=timeout
                )
            
            # Test connection
            print("Debug: Testing connection...")
            await self.logger.send_report(
                "qdrant_connection",
                "INFO",
                "Testing connection to Qdrant...",
                took=0
            )
            
            start_time = asyncio.get_event_loop().time()
            try:
                # Test connection with timeout
                await asyncio.wait_for(
                    self._test_connection(),
                    timeout=timeout
                )
                elapsed_time = round(asyncio.get_event_loop().time() - start_time, 2)
                print(f"Debug: Connection successful in {elapsed_time} seconds")
                await self.logger.send_report(
                    "qdrant_connection",
                    "INFO",
                    f"Successfully connected to Qdrant in {elapsed_time} seconds",
                    took=elapsed_time
                )
                return self.client
            except asyncio.TimeoutError:
                print("Debug: Connection timed out")
                await self.logger.send_report(
                    "qdrant_connection",
                    "ERROR",
                    f"Connection to Qdrant timed out after {timeout} seconds",
                    took=timeout
                )
                raise Exception(f"Connection to Qdrant timed out after {timeout} seconds")
            except Exception as e:
                print(f"Debug: Connection failed with error: {str(e)}")
                await self.logger.send_report(
                    "qdrant_connection",
                    "ERROR",
                    f"Failed to connect to Qdrant: {str(e)}",
                    took=0
                )
                raise Exception(f"Failed to connect to Qdrant: {str(e)}")
                
        except Exception as e:
            print(f"Debug: Connection failed with error: {str(e)}")
            await self.logger.send_report(
                "qdrant_connection",
                "ERROR",
                f"Failed to connect to Qdrant: {str(e)}",
                took=0
            )
            raise Exception(f"Failed to connect to Qdrant: {str(e)}")

    async def _test_connection(self):
        """Test the connection to Qdrant."""
        try:
            # Get collections to verify connection
            collections = await self.client.get_collections()
            return True
        except Exception as e:
            raise Exception(f"Connection test failed: {str(e)}")

    async def disconnect(self, client: Any) -> bool:
        """Disconnect from Qdrant"""
        try:
            if self.client:
                await self.client.close()
                self.client = None
            return True
        except Exception as e:
            print(f"Debug: Error during disconnect: {str(e)}")
            return False

    async def is_ready(self, client: Any) -> bool:
        """Check if Qdrant is ready."""
        try:
            if not self.client:
                return False
            await self.client.get_collections()
            return True
        except Exception as e:
            print(f"Debug: Error checking readiness: {str(e)}")
            return False

    async def create_collection(self, client: Any, collection_name: str, **kwargs) -> bool:
        """Create a new collection in Qdrant"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            # Check if collection exists
            collections = await self.client.get_collections()
            if any(col.name == collection_name for col in collections.collections):
                return True
            
            # Create collection with vector configuration
            vector_size = kwargs.get("vector_size", 384)  # Default to 384 for SentenceTransformers
            
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"Debug: Error creating collection: {str(e)}")
            return False

    async def delete_collection(self, client: Any, collection_name: str) -> bool:
        """Delete a collection from Qdrant"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            collections = await self.client.get_collections()
            if any(col.name == collection_name for col in collections.collections):
                await self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Debug: Error deleting collection: {str(e)}")
            return False

    async def insert_vectors(self, client: Any, collection_name: str, vectors: List[List[float]], metadata: List[Dict], **kwargs) -> List[str]:
        """Insert vectors into Qdrant collection"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            # Prepare points for insertion
            points = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                points.append(models.PointStruct(
                    id=i,
                    vector=vector,
                    payload=meta
                ))
            
            # Insert points
            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return [str(i) for i in range(len(points))]
        except Exception as e:
            print(f"Debug: Error inserting vectors: {str(e)}")
            return []

    def _convert_filters(self, filters: dict) -> Filter:
        """Convert standard filter format to Qdrant gRPC filter format"""
        if not filters:
            return None

        must_conditions = []
        should_conditions = []

        if "must" in filters:
            for condition in filters["must"]:
                if "key" in condition and "match" in condition:
                    match = condition["match"]
                    if "value" in match:
                        value = match["value"]
                        if isinstance(value, str):
                            must_conditions.append(
                                Condition(
                                    field=FieldCondition(
                                        key=condition["key"],
                                        match=Match(text=value)
                                    )
                                )
                            )
                        elif isinstance(value, bool):
                            must_conditions.append(
                                Condition(
                                    field=FieldCondition(
                                        key=condition["key"],
                                        match=Match(boolean=value)
                                    )
                                )
                            )
                        elif isinstance(value, int):
                            must_conditions.append(
                                Condition(
                                    field=FieldCondition(
                                        key=condition["key"],
                                        match=Match(integer=value)
                                    )
                                )
                            )
                        elif isinstance(value, float):
                            must_conditions.append(
                                Condition(
                                    field=FieldCondition(
                                        key=condition["key"],
                                        match=Match(double=value)
                                    )
                                )
                            )
                        else:
                            raise ValueError(f"Unsupported value type for gRPC filter: {type(value)}")
                    elif "range" in match:
                        must_conditions.append(
                            Condition(
                                field=FieldCondition(
                                    key=condition["key"],
                                    range=Range(**match["range"])
                                )
                            )
                        )
                    elif "any" in match:
                        for value in match["any"]:
                            if isinstance(value, str):
                                should_conditions.append(
                                    Condition(
                                        field=FieldCondition(
                                            key=condition["key"],
                                            match=Match(text=value)
                                        )
                                    )
                                )
                            elif isinstance(value, bool):
                                should_conditions.append(
                                    Condition(
                                        field=FieldCondition(
                                            key=condition["key"],
                                            match=Match(boolean=value)
                                        )
                                    )
                                )
                            elif isinstance(value, int):
                                should_conditions.append(
                                    Condition(
                                        field=FieldCondition(
                                            key=condition["key"],
                                            match=Match(integer=value)
                                        )
                                    )
                                )
                            elif isinstance(value, float):
                                should_conditions.append(
                                    Condition(
                                        field=FieldCondition(
                                            key=condition["key"],
                                            match=Match(double=value)
                                        )
                                    )
                                )
                            else:
                                raise ValueError(f"Unsupported value type in 'any' filter: {type(value)}")
        # Combine must and should conditions
        return Filter(must=must_conditions if must_conditions else None, should=should_conditions if should_conditions else None) if must_conditions or should_conditions else None

    async def search_vectors(self, client: Any, collection_name: str, query_vector: List[float], limit: int, **kwargs) -> List[Dict]:
        """Search for similar vectors in the collection"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            filters = kwargs.get('filters', None)
            
            # Convert filters to Qdrant format if provided
            query_filter = self._convert_filters(filters) if filters else None
            
            # Use query_points with proper parameters
            search_result = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=None,
                query_filter=query_filter
            )
            
            # Convert results to standard format
            results = []
            for point in search_result.points:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "metadata": point.payload
                })
            
            return results
            
        except Exception as e:
            print(f"Debug: Error searching vectors: {str(e)}")
            return []
    
    async def delete_vectors(self, client: Any, collection_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Qdrant collection"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            # Delete points
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[int(id) for id in vector_ids]
                )
            )
            return True
        except Exception as e:
            print(f"Debug: Error deleting vectors: {str(e)}")
            return False
    
    async def get_collection_info(self, client: Any, collection_name: str) -> Dict:
        """Get information about a Qdrant collection"""
        try:
            if not self.client:
                raise Exception("No active Qdrant client")
            
            # Get collection info
            collection = await self.client.get_collection(collection_name)
            
            # Get collection statistics
            stats = await self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "schema": collection.config,
                "count": stats.points_count
            }
        except Exception as e:
            print(f"Debug: Error getting collection info: {str(e)}")
            return {} 