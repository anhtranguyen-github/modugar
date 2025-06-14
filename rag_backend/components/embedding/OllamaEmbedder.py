import os
import requests
from wasabi import msg
import aiohttp
from urllib.parse import urljoin

from rag_backend.components.interfaces import Embedding
from rag_backend.components.types import InputConfig
from rag_backend.components.util import get_environment


class OllamaEmbedder(Embedding):

    def __init__(self):
        super().__init__()
        self.name = "Ollama"
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.description = f"Vectorizes documents and queries using Ollama. If your Ollama instance is not running on {self.url}, you can change the URL by setting the OLLAMA_URL environment variable."
        models = get_models(self.url)

        self.config = {
            "Model": InputConfig(
                type="dropdown",
                value=os.getenv("OLLAMA_EMBED_MODEL") or models[0],
                description=f"Select a installed Ollama model from {self.url}. You can change the URL by setting the OLLAMA_URL environment variable. ",
                values=models,
            ),
        }

    def get_vector_size(self, config: dict) -> int:
        """Get the vector size (dimension) for the configured model."""
        if "vector_dimension" in config:
            dim = config["vector_dimension"]
            if hasattr(dim, "value"):
                return int(dim.value)
            return int(dim)
        return 768  # fallback for Ollama

    async def vectorize(self, config: dict, content: list[str]) -> list[float]:

        model = config.get("Model").value

        data = {"model": model, "input": content}

        async with aiohttp.ClientSession() as session:
            async with session.post(urljoin(self.url, "/api/embed"), json=data) as response:
                response.raise_for_status()
                data = await response.json()
                embeddings = data.get("embeddings", [])
                return embeddings


def get_models(url: str):
    try:
        response = requests.get(urljoin(url, "/api/tags"))
        models = [model.get("name") for model in response.json().get("models")]
        if len(models) > 0:
            return models
        else:
            msg.info("No Ollama Model detected")
            return ["No Ollama Model detected"]
    except Exception as e:
        msg.info(f"Couldn't connect to Ollama {url}")
        return [f"Couldn't connect to Ollama {url}"]
