
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.31.0



docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

docker exec ollama ollama pull nomic-embed-text

