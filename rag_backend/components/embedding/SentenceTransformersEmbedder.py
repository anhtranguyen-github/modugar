from rag_backend.components.interfaces import Embedding
from rag_backend.components.types import InputConfig

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    pass


class SentenceTransformersEmbedder(Embedding):
    """
    SentenceTransformersEmbedder base class for Verba.
    """

    def __init__(self):
        super().__init__()
        self.name = "SentenceTransformers"
        self.requires_library = ["sentence_transformers"]
        self.description = "Embeds and retrieves objects using SentenceTransformer"
        self.config = {
            "Model": InputConfig(
                type="dropdown",
                value="all-MiniLM-L6-v2",
                description="Select an HuggingFace Embedding Model",
                values=[
                    "all-MiniLM-L6-v2",
                    "mixedbread-ai/mxbai-embed-large-v1",
                    "all-mpnet-base-v2",
                    "BAAI/bge-m3",
                    "all-MiniLM-L12-v2",
                    "paraphrase-MiniLM-L6-v2",
                ],
            ),
        }

    async def vectorize(self, config: dict, content: list[str]) -> list[float]:
        try:
            model_name = config["Model"].value if hasattr(config["Model"], "value") else config["Model"]
            model = SentenceTransformer(model_name)
            embeddings = model.encode(content).tolist()
            return embeddings
        except Exception as e:
            raise Exception(f"Failed to vectorize chunks: {str(e)}")

    def get_vector_size(self, config: dict) -> int:
        """Get the vector size (dimension) for the configured model."""
        if "vector_dimension" in config:
            dim = config["vector_dimension"]
            if hasattr(dim, "value"):
                return int(dim.value)
            return int(dim)
        try:
            model_name = config["Model"].value if hasattr(config["Model"], "value") else config["Model"]
            model = SentenceTransformer(model_name)
            return model.get_sentence_embedding_dimension()
        except Exception:
            return 384  # fallback
