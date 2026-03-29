# Optional sandbox: `docker build -t autoresearcher-sandbox:latest -f docker/Dockerfile .`
FROM python:3.11-slim
WORKDIR /work
RUN pip install --no-cache-dir numpy pandas scikit-learn pyyaml pydantic httpx anthropic langgraph langchain-core
