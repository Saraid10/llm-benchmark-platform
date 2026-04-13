FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install in exact order - huggingface_hub FIRST to prevent gradio upgrading it
RUN pip install --no-cache-dir huggingface_hub==0.23.4
RUN pip install --no-cache-dir "gradio==4.26.0"
RUN pip install --no-cache-dir \
    duckdb \
    pandas \
    numpy \
    pydantic \
    plotly \
    psutil \
    rich \
    python-dotenv

COPY . .
RUN mkdir -p data/raw data/versions
ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "app.py"]
