Models downloaded into: /home/nat7/.cache/docling/models.

 Docling can now be configured for running offline using the local artifacts.

 Using the CLI: `docling --artifacts-path=/home/nat7/.cache/docling/models FILE` 
 Using Python: see the documentation at <https://docling-project.github.io/docling/usage>.


docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama


docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec ollama ollama pull nomic-embed-text
