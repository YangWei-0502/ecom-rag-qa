FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/chroma_db data/uploads logs && python scripts/init_db.py

EXPOSE 8512

# 容器内访问宿主机 Ollama 时，OLLAMA_BASE_URL 设为
# http://host.docker.internal:11434/v1
CMD ["python", "-m", "streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8512"]
