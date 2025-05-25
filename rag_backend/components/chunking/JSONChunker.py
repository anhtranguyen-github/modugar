import contextlib

import json

with contextlib.suppress(Exception):
    from langchain_text_splitters import (
        RecursiveJsonSplitter,
    )

from rag_backend.components.chunk import Chunk
from rag_backend.components.interfaces import Chunker
from rag_backend.components.document import Document
from rag_backend.components.types import InputConfig
from rag_backend.components.interfaces import Embedding


class JSONChunker(Chunker):
    """
    JSONChunker for Verba using LangChain.
    """

    def __init__(self):
        super().__init__()
        self.name = "JSON"
        self.requires_library = ["langchain_text_splitters"]
        self.description = "Split json files using LangChain"
        self.config = {
            "Chunk Size": InputConfig(
                type="number",
                value=500,
                description="Choose how many characters per chunks",
                values=[],
            ),
        }

    async def chunk(
        self,
        config: dict,
        documents: list[Document],
        embedder: Embedding | None = None,
        embedder_config: dict | None = None,
    ) -> list[Document]:

        units = int(config["Chunk Size"].value)

        text_splitter = RecursiveJsonSplitter(max_chunk_size=units)

        for document in documents:

            json_obj = json.loads(document.content)

            # Skip if document already contains chunks
            if len(document.chunks) > 0:
                continue

            char_end_i = -1
            for i, chunk in enumerate(text_splitter.split_text(json_obj)):

                char_start_i = char_end_i + 1
                char_end_i = char_start_i + len(chunk)

                document.chunks.append(
                    Chunk(
                        content=chunk,
                        chunk_id=i,
                        start_i=None,  # not implemented as the splitter modifies the outputs
                        end_i=None,
                        content_without_overlap=chunk,
                    )
                )

        return documents
