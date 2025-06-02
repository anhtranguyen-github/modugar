from rag_backend.components.interfaces import Retriever
from rag_backend.components.types import InputConfig


class WindowRetriever(Retriever):
    """
    WindowRetriever that retrieves chunks and their surrounding context depending on the window size.
    """

    def __init__(self):
        super().__init__()
        self.description = "Retrieve relevant chunks from vector store"
        self.name = "Advanced"
        self._vector_store_manager = None

        self.config["Search Mode"] = InputConfig(
            type="dropdown",
            value="Hybrid Search",
            description="Switch between search types.",
            values=["Hybrid Search"],
        )
        self.config["Limit Mode"] = InputConfig(
            type="dropdown",
            value="Autocut",
            description="Method for limiting the results. Autocut decides automatically how many chunks to retrieve, while fixed sets a fixed limit.",
            values=["Autocut", "Fixed"],
        )
        self.config["Limit/Sensitivity"] = InputConfig(
            type="number",
            value=1,
            description="Value for limiting the results. Value controls Autocut sensitivity and Fixed Size",
            values=[],
        )
        self.config["Chunk Window"] = InputConfig(
            type="number",
            value=1,
            description="Number of surrounding chunks of retrieved chunks to add to context",
            values=[],
        )
        self.config["Threshold"] = InputConfig(
            type="number",
            value=80,
            description="Threshold of chunk score to apply window technique (1-100)",
            values=[],
        )

    @property
    def vector_store_manager(self):
        if self._vector_store_manager is None:
            from rag_backend.components.managers import VectorStoreManager
            self._vector_store_manager = VectorStoreManager()
        return self._vector_store_manager

    async def retrieve(
        self,
        client,
        query,
        vector,
        config,
        embedder,
        labels,
        document_uuids,
        collection_name: str,
    ):
        # Initialize vector store manager if not already set
        if self._vector_store_manager is None:
            from rag_backend.components.managers import VectorStoreManager
            self._vector_store_manager = VectorStoreManager()

        search_mode = config["Search Mode"].value
        limit_mode = config["Limit Mode"].value
        limit = int(config["Limit/Sensitivity"].value)

        window = max(0, min(10, int(config["Chunk Window"].value)))
        window_threshold = max(0, min(100, int(config["Threshold"].value)))
        window_threshold /= 100

        if search_mode == "Hybrid Search":
            # Use vector store manager for search
            search_results = await self.vector_store_manager.search_vectors(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit,
                filters={
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"any": document_uuids}
                        }
                    ]
                } if document_uuids else None
            )
            
            # Convert search results to chunks format and deduplicate by chunk_id
            chunks = []
            seen_chunk_ids = set()
            for result in search_results:
                chunk_id = result["metadata"]["chunk_id"]
                if chunk_id not in seen_chunk_ids:
                    chunks.append({
                        "properties": {
                            "doc_uuid": result["metadata"]["doc_id"],
                            "chunk_id": chunk_id,
                            "content": result["metadata"]["text"]
                        },
                        "uuid": result["id"],
                        "metadata": {"score": result["score"]}
                    })
                    seen_chunk_ids.add(chunk_id)

        if len(chunks) == 0:
            return ([], "We couldn't find any chunks to the query")

        # Group Chunks by document and sum score
        doc_map = {}
        scores = [0]
        for chunk in chunks:
            if chunk["properties"]["doc_uuid"] not in doc_map:
                # Get document info from vector store
                doc_info = await self.vector_store_manager.get_collection_info(collection_name)
                if doc_info is None:
                    continue
                doc_map[chunk["properties"]["doc_uuid"]] = {
                    "title": doc_info.get("name", "Unknown"),
                    "chunks": [],
                    "score": 0,
                    "metadata": doc_info.get("metadata", {}),
                }
            
            doc_map[chunk["properties"]["doc_uuid"]]["chunks"].append({
                "uuid": str(chunk["uuid"]),
                "score": chunk["metadata"]["score"],
                "chunk_id": chunk["properties"]["chunk_id"],
                "content": chunk["properties"]["content"],
            })
            doc_map[chunk["properties"]["doc_uuid"]]["score"] += chunk["metadata"]["score"]
            scores.append(chunk["metadata"]["score"])
        min_score = min(scores)
        max_score = max(scores)

        def normalize_value(value, max_value, min_value):
            return (value - min_value) / (max_value - min_value)

        def generate_window_list(value, window):
            value = int(value)
            window = int(window)
            return [i for i in range(value - window, value + window + 1) if i != value]

        documents = []
        context_documents = []

        for doc in doc_map:
            additional_chunk_ids = set()  # Use set for automatic deduplication
            chunks_above_threshold = 0
            for chunk in doc_map[doc]["chunks"]:
                normalized_score = normalize_value(
                    float(chunk["score"]), float(max_score), float(min_score)
                )
                if window_threshold <= normalized_score:
                    chunks_above_threshold += 1
                    additional_chunk_ids.update(generate_window_list(
                        chunk["chunk_id"], window
                    ))

            if additional_chunk_ids:
                # Get additional chunks using vector store manager
                additional_results = await self.vector_store_manager.search_vectors(
                    collection_name=collection_name,
                    query_vector=vector,
                    limit=len(additional_chunk_ids),
                    filters={
                        "must": [
                            {
                                "key": "doc_id",
                                "match": {"value": doc}
                            },
                            {
                                "key": "chunk_id",
                                "match": {"any": list(additional_chunk_ids)}
                            }
                        ]
                    }
                )
                
                existing_chunk_ids = {
                    chunk["chunk_id"] for chunk in doc_map[doc]["chunks"]
                }
                
                for result in additional_results:
                    chunk_id = result["metadata"]["chunk_id"]
                    if chunk_id not in existing_chunk_ids:
                        doc_map[doc]["chunks"].append({
                            "uuid": str(result["id"]),
                                "score": 0,
                            "chunk_id": chunk_id,
                            "content": result["metadata"]["text"],
                        })
                        existing_chunk_ids.add(chunk_id)

            # Sort chunks by chunk_id to ensure consistent order
            doc_map[doc]["chunks"].sort(key=lambda x: x["chunk_id"])

            _chunks = [
                {
                    "uuid": str(chunk["uuid"]),
                    "score": chunk["score"],
                    "chunk_id": chunk["chunk_id"],
                    "embedder": embedder,
                }
                for chunk in doc_map[doc]["chunks"]
            ]
            context_chunks = [
                {
                    "uuid": str(chunk["uuid"]),
                    "score": chunk["score"],
                    "content": chunk["content"],
                    "chunk_id": chunk["chunk_id"],
                    "embedder": embedder,
                }
                for chunk in doc_map[doc]["chunks"]
            ]

            documents.append({
                    "title": doc_map[doc]["title"],
                "chunks": _chunks,
                    "score": doc_map[doc]["score"],
                    "metadata": doc_map[doc]["metadata"],
                    "uuid": str(doc),
            })

            context_documents.append({
                    "title": doc_map[doc]["title"],
                "chunks": context_chunks,
                    "score": doc_map[doc]["score"],
                    "uuid": str(doc),
                    "metadata": doc_map[doc]["metadata"],
            })

        sorted_context_documents = sorted(
            context_documents, key=lambda x: x["score"], reverse=True
        )
        sorted_documents = sorted(documents, key=lambda x: x["score"], reverse=True)

        context = self.combine_context(sorted_context_documents)
        return (sorted_documents, context)

    def combine_context(self, documents: list[dict]) -> str:
        context = ""
        for document in documents:
            context += f"Document Title: {document['title']}\n"
            if len(document["metadata"]) > 0:
                context += f"Document Metadata: {document['metadata']}\n"
            for chunk in document["chunks"]:
                context += f"Chunk: {int(chunk['chunk_id'])+1}\n"
                if chunk["score"] > 0:
                    context += f"High Relevancy: {chunk['score']:.2f}\n"
                context += f"{chunk['content']}\n"
            context += "\n\n"
        return context
