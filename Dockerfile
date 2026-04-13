FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install exact known-good versions in correct order
RUN pip install --no-cache-dir \
    jinja2==3.1.2 \
    huggingface_hub==0.23.4 \
    gradio==4.26.0 \
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

CMD ["python", "src/ui/app_hf.py"]
