FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install exact versions known to work together
RUN pip install --no-cache-dir \
    "jinja2==3.1.4" \
    "huggingface_hub==0.23.4" \
    "gradio==4.26.0" \
    "starlette==0.27.0" \
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
